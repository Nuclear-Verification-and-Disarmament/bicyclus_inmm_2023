import json
import math
from collections import namedtuple
from pathlib import Path
from shutil import copy

import numpy as np
import pymc as pm
from aesara.tensor import as_tensor_variable
from bicyclus import blackbox, cyclus_db, util

import cyclus_input


# Fix seed to ensure reproducibility. RNG_SEED may be updated in case of
# running calculations on multiple instances. After that, the RNG is set.
RNG = None
RNG_SEED = 12345

# namedtuple is used to facilitate extracting parameters and calculating the
# likelihood. This variable might change later depending on which outputs are
# extracted.
SimulationOutput = namedtuple("SimulationOutput", ("cs137_mass", "parameters"))


class PakistanCyclusModel(blackbox.CyclusCliModel):
    """This class handles all interaction with Cyclus.

    Notably, it generates the input files, starts simulations and reads from
    the output files.
    """

    def __init__(self, true_parameters, sampled_parameters):
        self.true_parameters = true_parameters
        self.sampled_parameters = sampled_parameters
        self.current_parameters = None

        super().__init__()

    def mutate(self, sample=None):
        """Mutate the model, i.e. update its parameters.

        Parameters
        ----------
        sample : list
            Contains parameters corresponding to the parameters in
            `sampled_parameters`. The order must be identical to the
            alphabetically sorted order of keys in `sampled_parameters`.
        """
        # Get ground truth simulation result.
        if sample is None:
            self.mut_model = cyclus_input.simulation(parameters=self.true_parameters)
            return

        # Alphabetical order, see model. Only copy keys, as sampled_parameters
        # contain the sampling range.
        parameters = {k: None for k in self.sampled_parameters}
        # Unpack alphabetically sorted model parameters.
        for i, k in enumerate(sorted(self.sampled_parameters.keys())):
            parameters[k] = sample[i]
        util.log_print(f"Mutating: using parameters {parameters}")

        self.mut_model = cyclus_input.simulation(parameters=parameters)
        self.current_parameters = parameters

    def result(self):
        """Extract the relevant output from the last simulation."""
        fname = self.last_sqlite_file
        spent_fuel_composition, spent_fuel_qty = cyclus_db.run_with_conn(
            fname,
            cyclus_db.extract_transaction_composition,
            {"agent_name": "FinalWasteSink"},
        )
        cs137_mass = spent_fuel_composition[551370000] * spent_fuel_qty

        return SimulationOutput(
            cs137_mass=cs137_mass,
            parameters=self.current_parameters,
        )


class PakistanLikelihood(blackbox.LikelihoodFunction):
    """Class to extract simulation results and calculate the likelihood."""

    def __init__(self, truth: SimulationOutput, rel_sigma=0.5):
        """Create a IsotopeLikelihood object.

        Parameters
        ----------
        truth : SimulationOutput,
            The ground truth, i.e., simulation results using the true
            parameters.

        rel_sigma : float
            Relative sigma *in percent*.
        """
        self.truth = truth
        self.rel_sigma = rel_sigma  # in percent.

    def log_likelihood(self, output: SimulationOutput):
        """Calculate the loglikelihood."""

        def abs_sigma(x):
            """Convert relative sigma to absolute sigma."""
            return x * self.rel_sigma / 100

        def std_normal(x):
            """Height of standard normal distribution at x."""
            return math.exp(-(x**2) / 2) / (2 * math.pi) ** 0.5

        total_llk = 0
        for contrib in SimulationOutput._fields:
            if contrib.lower() == "parameters":
                continue
            # Normalise the differences to make all likelihood contributions
            # comparable.
            output_contrib = getattr(output, contrib)
            true_contrib = getattr(self.truth, contrib)
            normalised = (output_contrib - true_contrib) / abs_sigma(true_contrib)
            lik = std_normal(normalised)  # Likelihood
            try:
                loglik = math.log(lik)
            except ValueError as e:
                if lik < 1e-30:  # Arbitrarily chosen very small value.
                    loglik = -np.inf
                else:
                    raise e

            total_llk += loglik
            util.log_print(f"  loglikehood contribution {contrib}: {loglik:.5e}")

        util.log_print(f"Total loglikelihood: {total_llk:.5e}")

        # Variable has to be returned as an array.
        return np.array(total_llk)


class SaveCyclusSample:
    """Class that pymc.sample uses as callback function."""

    def __init__(self, args, cyclus_model):
        self.cyclus_model = cyclus_model
        self.run = args.run
        self.saved_files = 0

        self.sqlite_outdir = Path(args.output_path, "sqlite_files")
        self.sqlite_outdir.mkdir(mode=0o760, parents=True, exist_ok=True)

    def __call__(self, trace, draw):
        """Callback function for sampling process."""
        if draw.tuning:
            # We do not want to save tuning steps.
            return

        sqlite_fname = "cyclus_output_{}_{}_{}.sqlite".format(
            self.run, util.task_identifier(), self.saved_files
        )
        sqlite_fname = self.sqlite_outdir / sqlite_fname
        try:
            copy(self.cyclus_model.last_sqlite_file, sqlite_fname)
            util.log_print(f"Saved Cyclus output {self.saved_files}")
        except FileNotFoundError as e:
            util.log_print(f"Error saving Cyclus output {sqlite_fname}: {e}")

        self.saved_files += 1


class SimpleCallback:
    """Basic callback class that shows the progress during sampling."""

    def __init__(self):
        self.tuning_step = 0
        self.sample_step = 0

    def __call__(self, trace, draw):
        """Callback function for sampling process."""
        if draw.tuning:
            util.log_print(f"Tuning sample {self.tuning_step}")
            self.tuning_step += 1
            return

        util.log_print(f"Sample {self.sample_step}")
        self.sample_step += 1


def model(args):
    """Construct the model with sampled random initial values.

    Parameters
    ----------
    args : arguments passed from the argparser

    Returns
    -------
    pymc_model : pm.Model
    initvals : array_like
    cyclus_model : PakistanCyclusModel
    """
    # Read prior distributions and groundtruths from files.
    with open(args.sample_parameters_file, "r") as f:
        sample_parameters = json.load(f)
    with open(args.true_parameters_file, "r") as f:
        true_parameters = json.load(f)

    cyclus_model = PakistanCyclusModel(true_parameters, sample_parameters)
    groundtruth = cyclus_model.run_groundtruth()
    util.log_print(f"Ground truth parameters are: {groundtruth}")

    loglikelihood_op = blackbox.CyclusLogLikelihood(
        PakistanLikelihood(groundtruth, rel_sigma=args.rel_sigma),
        cyclus_model,
        memoize=True,
    )

    util.log_print("Building PyMC model.")
    util.log_print(
        "Sampling variables as follows:",
        [f"{k} => {v}" for (k, v) in sample_parameters.items()],
    )
    util.log_print(
        "The true parameters are:",
        [f"{k} => {v}" for (k, v) in true_parameters.items()],
    )

    with pm.Model() as pymc_model:
        # Transform the priors from the .json file to PyMC distributions.
        pymc_priors = {
            name: util.sampling_parameter_to_pymc(name, prior)
            for name, prior in sample_parameters.items()
        }

        util.log_print("Model variables:", pymc_priors)

        pm.Potential(
            "observed",
            loglikelihood_op(
                as_tensor_variable([pymc_priors[k] for k in sorted(pymc_priors.keys())])
            ),
        )

        # Generate the initial values.
        initvals = util.generate_start_values(sample_parameters, RNG, args.chains)

    return pymc_model, initvals, cyclus_model


def sample(args, pymc_model, initvals, cyclus_model):
    """Sample the random variables, generate and save the trace.

    Parameters
    ----------
    args : arguments passed from the argparser
    pymc_model : pm.Model
    initvals : array_like
    cyclus_model : PakistanCyclusModel
    """
    with pymc_model:
        # Set algorithm used as Step function.
        if args.algorithm == "default":
            algorithm = pm.Slice()
        else:
            try:
                algorithm = pm.step_methods.__dict__[args.algorithm]()
            except KeyError as e:
                msg = (
                    "--algorithm must be one of the methods defined by "
                    "PyMC, see "
                    "https://www.pymc.io/projects/docs/en/stable/api/samplers.html."
                    "Note that PyMC capitalises the first letter (e.g., "
                    "'Metropolis' instead of 'metropolis'). Also note "
                    "that not all algorithms may work here, some may need "
                    "changes in the code."
                )
                raise KeyError(msg) from e

        if args.store_sqlite:
            callback = SaveCyclusSample(args, cyclus_model)
        else:
            callback = SimpleCallback()

        util.log_print("Start of sampling")
        util.log_print(
            f"Sampling {args.tune} tuning steps and {args.samples} samples "
            f"using {args.algorithm}, initial parameters {initvals}"
        )
        util.log_print(f"Using callback function: {args.store_sqlite}")
        inference_data = pm.sample(
            draws=args.samples,
            tune=args.tune,
            step=algorithm,
            chains=args.chains,
            cores=args.cores,
            initvals=initvals,
            return_inferencedata=True,
            compute_convergence_checks=False,
            progressbar=False,
            random_seed=RNG,
            callback=callback,
        )
        util.save_trace(args, inference_data)

    util.log_print("Sampling finished!")


def main():
    """Main entry point"""
    global RNG, RNG_SEED

    parser = util.ReconstructionParser().get_args(parsed=False)
    parser.add_argument(
        "--store-sqlite",
        action="store_true",
        help=(
            "If set, then the Cyclus .sqlite output of each (accepted) sample is stored."
        ),
    )
    args = parser.parse_args()
    args.iterations = 0
    if args.chains != 1:
        msg = (
            "Sampling more than one chain will result in errors in the "
            "callback function. Please run multiple sampling processes in "
            "parallel instead of using the 'chains' argument."
        )
        raise RuntimeError(msg)

    # Open stream to log file.
    print(f"Using debug mode: {args.debug}", flush=True)
    print(f"Writing log to {args.log_path}", flush=True)
    util.write_to_log_file(run=args.run, outpath=args.log_path, debug=args.debug)
    util.log_print(f"Sampling run {args.run}, task {args.index}")
    util.log_print(f"Running with arguments: {args}")

    # Set seed, see beginning of file for more explications.
    RNG_SEED += args.index
    RNG = np.random.default_rng(seed=RNG_SEED)
    util.log_print(f"RNG: {RNG}")
    util.log_print(f"RNG seed: {RNG_SEED}")

    # Initialise PyMC model.
    pymc_model, initvals, cyclus_model = model(args)
    # Let's go!
    sample(args, pymc_model, initvals, cyclus_model)


if __name__ == "__main__":
    main()
