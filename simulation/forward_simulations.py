from bicyclus.blackbox import CyclusForwardModel
from bicyclus.util import ForwardSimulationParser, log

import cyclus_input


RNG_SEED = 12345


class ForwardSimulator(CyclusForwardModel):
    """ForwardSimulator generates the input file."""

    def generate_input_file(self, sample):
        return cyclus_input.simulation(parameters=sample)


def main():
    """Initiate and run the simulations."""
    parser = ForwardSimulationParser()
    args = parser.get_args()
    base_fnames = f"{args.run}_{log.task_identifier()}"
    # Update seed to prevent using the same seed when running multiple sampling
    # processes in parallel.
    seed = RNG_SEED + args.index

    # Initiate log file.
    log.write_to_log_file(run=base_fnames, outpath=args.log_path)
    log.log_print(f"Forward sampling run {args.run}, task {args.index}")
    log.log_print(f"Running with arguments: {args}")

    # Initiate simulator, draw parameter samples and run the simulations.
    simulator = ForwardSimulator(
        input_params_fname=args.sample_parameters_file,
        n_samples_exponent=args.n_samples_exponent,
        seed=seed,
        data_output_dir=args.output_path,
        log_output_dir=args.log_path,
        output_fnames=base_fnames,
    )
    simulator.run_simulations()


if __name__ == "__main__":
    main()
