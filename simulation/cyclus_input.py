#!/usr/bin/env python3

"""A script generating a JSON input file for Cyclus."""


import json
from datetime import datetime
from subprocess import run, PIPE, STDOUT
from sys import stdout

from bicyclus.util import CyclusRunParser

import pakistan_data
from cyclus_input_helper_functions import (
    yearly_rates_to_sim_steps,
    date_to_sim_step,
    add_reprocessing,
    add_sink,
    add_source,
    add_storage,
    add_uranium_recipe,
)
from global_parameters import PARAMS
from reactor_input_file_writer import KhushabReactor


def main():
    """Main function, only meant for testing and debugging.

    This function might get deleted later and is quite hacky. The actual
    function used during forward and reconstruction simulations is
    `simulation`.
    """
    p = CyclusRunParser()
    p.parser.add_argument(
        "--parameters",
        default=None,
        type=str,
        help="A JSON file containing additional that may "
        "override parameters in PARAMS.",
    )
    args = p.get_args()
    additional_params = {}
    if args.parameters is not None:
        with open(args.parameters, "r") as f:
            additional_params = json.load(f)

    # Run a Cyclus simulation as subprocess, store input, output, and STDOUT
    # and STDERR in respective files.
    if args.debug_mode:
        if not args.outfile:
            raise RuntimeError("No filename for outputfile given!")

        sim = simulation(name=args.name, parameters=additional_params)
        fname = args.outfile.split(".")[0]
        in_fname = fname + "_input.json"
        log_fname = fname + "_log.txt"
        out_fname = fname + ".sqlite"
        with open(in_fname, "w") as f:
            json.dump(sim, f, indent=1)

        # Remove old sqlite file as sqlite files append instead of overwrite.
        run(["rm", out_fname])

        save_log = 1
        if save_log:
            result = run(
                ["cyclus", "-i", in_fname, "-o", out_fname, "-v", "11"],
                stdout=PIPE,
                stderr=STDOUT,
                encoding="utf-8",
            )
            with open(log_fname, "w") as f:
                f.write(result.stdout)
        else:
            result = run(["cyclus", "-i", in_fname, "-o", out_fname, "-v", "11"])
        return

    json.dump(
        simulation(name=args.name, parameters=additional_params), stdout, indent=2
    )
    return


def simulation(name="Pakistan", parameters=None):
    """Generate and save a Cyclus input file."""
    if parameters is not None:
        PARAMS.update(parameters)
    PARAMS.update(name=name)
    if PARAMS["global_capacity_factor"] is not None:
        for i in range(1, 5):
            key = f"khushab{i}_capacity_factor"
            if key not in PARAMS:
                msg = f"Cannot find key '{key}' in PARAMS"
                raise RuntimeError(msg)
            PARAMS[key] = PARAMS["global_capacity_factor"]
    PARAMS.update(
        reactor_list=[
            KhushabReactor(
                name=f"Khushab{n}",
                power=PARAMS[f"khushab{n}_power"],
                burnup=PARAMS[f"khushab{n}_burnup"],
                capacity_factor=PARAMS[f"khushab{n}_capacity_factor"],
                deployment_year=PARAMS[f"khushab{n}_deployment"],
                fuel_pref=PARAMS["khushab_fuel_pref"],
                core_mass=PARAMS["core_mass"],
                simulation_ts=PARAMS["dt"],
            )
            for n in range(1, 5)
        ]
    )

    if not PARAMS["add_waste_sink_in_last_step"]:
        msg = (
            f"PARAMS['add_waste_sink_in_last_step'] is set to "
            f"{PARAMS['add_waste_sink_in_last_step']}. This *will* "
            "interfere and possibly crash the sampling and inference "
            "process. If you still want to continue, you need to comment "
            "out this error."
        )
        raise ValueError(msg)

    simulation = {
        "simulation": {
            **archetypes(),
            **commodities(),
            **control(),  # Must be called *before* facilities() and region().
            **facilities(),  # Must be called *before* region().
            **recipes(),
            **region(),
        }
    }
    return simulation


def archetypes():
    """Return the 'archetypes' part of the Cyclus input file as a dict."""
    archetypes_by_library = {
        "agents": ["NullRegion"],
        "cycamore": [
            "DeployInst",
            "Reactor",
            "Separations",
            "Sink",
            "Source",
            "Storage",
        ],
        "flexicamore": [
            "FlexibleEnrichment",
            "FlexibleSource",
        ],
    }
    archetypes_dict = {
        "archetypes": {
            "spec": [
                {"lib": lib, "name": archetype}
                for lib, archetypes in archetypes_by_library.items()
                for archetype in archetypes
            ]
        }
    }
    return archetypes_dict


def commodities():
    """Return the 'commodity' part of the Cyclus input file as a dict."""
    commods = [
        "NaturalU",
        "WeapongradeU",
        "DepletedU",
        "SpentFuel",
        "SeparatedPu",
        "SeparatedU",
        "SeparatedWaste",
        "MinedU",
        "FreshFuel",
        "FinalWaste",
    ]

    return {"commodity": [{"name": c, "solution_priority": 1} for c in commods]}


def control():
    """Return the 'control' part of the Cyclus input file as a dict."""
    controls = [
        "decay",
        "dt",
        "explicit_inventory",
        "explicit_inventory_compact",
        "simhandle",
        "startmonth",
        "startyear",
    ]
    control_dict = {"control": {c: PARAMS[c] for c in controls}}

    PARAMS["startdate"] = datetime(PARAMS["startyear"], 1, 1)
    PARAMS["duration"] = date_to_sim_step(PARAMS["endyear"])
    control_dict["control"]["duration"] = PARAMS["duration"]

    if PARAMS["solver"].lower() == "greedy":
        control_dict["control"]["solver"] = {"config": {"greedy": None}}
    elif PARAMS["solver"].lower() in ("coinor", "coin-or"):
        control_dict["control"]["solver"] = (
            {"config": {"coin-or": {"verbose": False}}},
        )
    else:
        msg = (
            "Invalid value used in `PARAMS['solver']`!"
            f"Currently, it is set to {PARAMS['solver']}."
        )
        raise ValueError(msg)

    return control_dict


def facilities():
    """Return the 'facilities' part of the Cyclus input file as a dict."""
    # Read in data.
    mine_production = pakistan_data.natural_u_production_df(
        PARAMS["current_mining_production"]
    )
    swu_capacity = pakistan_data.swu_df(
        PARAMS["swu_increase2"], PARAMS["swu_increase3"]
    )

    # Convert data into appropriate rates.
    mine_throughput_vals, mine_throughput_times = yearly_rates_to_sim_steps(
        mine_production["mass"], mine_production["year"]
    )
    swu_capacity_vals, swu_capacity_times = yearly_rates_to_sim_steps(
        swu_capacity["swu"], swu_capacity["year"]
    )

    # Safety check to prevent SWUs larger than the equivalent feed available.
    max_swu_per_ts = PARAMS["max_swu_per_day"] * PARAMS["dt"] / 86400
    if max(swu_capacity_vals) > max_swu_per_ts:
        raise ValueError(
            f"Maximum SWU per timestep larger than limit of {max_swu_per_ts} "
            "SWU per timestep. The feed inventory's capacity may be too small "
            "for such a large SWU."
        )
    max_feed_inv = PARAMS["enrichment_max_feed_inv"] * PARAMS["dt"] / 86400

    enrich_feed_commods = ["NaturalU", "SeparatedU"]
    enrich_feed_prefs = [2, 1]

    facilities_dict = {
        "facility": [
            {
                "name": "UraniumMine",
                "config": {
                    "FlexibleSource": {
                        "out_commod": "MinedU",
                        "out_recipe": "NaturalURecipe",
                        "inventory_size": 1e299,
                        "throughput_times": {"val": mine_throughput_times},
                        "throughput_vals": {"val": mine_throughput_vals},
                    }
                },
            },
            {
                "name": "EnrichmentFacility",
                "config": {
                    "FlexibleEnrichment": {
                        "feed_commods": {"val": enrich_feed_commods},
                        "feed_commod_prefs": {"val": enrich_feed_prefs},
                        "product_commod": "WeapongradeU",
                        "tails_commod": "DepletedU",
                        "tails_assay": PARAMS["DU_enrichment_atom"],
                        "max_feed_inventory": max_feed_inv,
                        "max_enrich": 1.0,
                        "order_prefs": False,
                        "swu_capacity_times": {"val": swu_capacity_times},
                        "swu_capacity_vals": {"val": swu_capacity_vals},
                    }
                },
            },
            add_source(
                "InitialNUStockpile",
                "MinedU",
                recipe="NaturalURecipe",
                source_kwargs={
                    "throughput": 1e299,
                    "inventory_size": PARAMS["initial_u_stockpile"],
                },
                facility_kwargs={"lifetime": 2},
            ),
            add_sink("SeparatedPu", recipe=""),
            add_sink("DepletedU"),
            # This storage facility is necessary to act as a buffer for the uranium
            # mine to prevent loss of mined U. The in- and out-commodities have to
            # be named differently to prevent trading with themselves.
            add_storage(
                "NaturalUStorage", "MinedU", "NaturalU", recipe="NaturalURecipe"
            ),
            # This storage is necessary to accumulate enough uranium s.t. the
            # reactor can obtain it as one batch in one resource exchange.
            add_storage(
                "FreshFuelStorage",
                "NaturalU",
                "FreshFuel",
                recipe="NaturalURecipe",
                max_inv_size=4 * PARAMS["core_mass"],
                in_commod_prefs={"val": [PARAMS["khushab_fuel_pref"]]},
            ),
            add_sink("WeapongradeU"),
            add_reprocessing(
                "ReprocessingFacility", "SpentFuel", PARAMS["separation_eff"]
            ),
        ]
    }
    for reactor in PARAMS["reactor_list"]:
        facilities_dict["facility"].append(
            reactor.create_facility(initial_year=control()["control"]["startyear"])
        )

    if PARAMS["add_waste_sink_in_last_step"]:
        facilities_dict["facility"].append(
            add_storage(
                "SeparatedWasteStorage", "SeparatedWaste", "FinalWaste", recipe=""
            )
        )
        facilities_dict["facility"].append(add_sink("FinalWaste", recipe=""))
    else:
        facilities_dict["facility"].append(add_sink("SeparatedWaste", recipe=""))

    return facilities_dict


def institution():
    """Return the 'institution' part of the Cyclus input file as a dict."""
    deployed_prototypes = [r.name for r in PARAMS["reactor_list"]]
    build_times = [date_to_sim_step(r.deployment_year) for r in PARAMS["reactor_list"]]

    # Add waste sink in the last time step (hack to force decay calculations).
    if PARAMS["add_waste_sink_in_last_step"]:
        if PARAMS["duration"] is None:
            control()
        deployed_prototypes.append("FinalWasteSink")
        build_times.append(PARAMS["duration"] - 1)

    if any(bt < 1 for bt in build_times):
        msg = (
            f"Deployed prototypes are {deployed_prototypes}."
            f"Corresponding build times are {build_times}."
            "All build_times must be strictly larger than 0!"
        )
        raise ValueError(msg)

    initial_facilities = [
        fac["name"]
        for fac in facilities()["facility"]
        if fac["name"] not in deployed_prototypes
    ]
    d = {
        "institution": [
            {
                "name": "PakistanAtomicEnergyCommission",
                "config": {
                    "DeployInst": {
                        "prototypes": {"val": deployed_prototypes},
                        "build_times": {"val": build_times},
                        "n_build": {"val": [1] * len(deployed_prototypes)},
                    }
                },
                "initialfacilitylist": {
                    "entry": [
                        {"number": 1, "prototype": facility}
                        for facility in initial_facilities
                    ]
                },
            }
        ]
    }
    return d


def recipes():
    """Return the 'recipe' part of the Cyclus input file as a dict.

    Compositions do not need be normalised to 1, Cyclus does this.
    """
    natural_uranium = add_uranium_recipe(
        "NaturalURecipe", "mass", PARAMS["NU_assay_mass"]
    )
    weapongrade_uranium = add_uranium_recipe(
        "WeapongradeURecipe", "atom", PARAMS["WGU_enrichment_atom"]
    )
    depleted_uranium = add_uranium_recipe(
        "DepletedURecipe", "atom", PARAMS["DU_enrichment_atom"]
    )

    recipes_dict = {
        "recipe": [
            natural_uranium,
            weapongrade_uranium,
            depleted_uranium,
        ]
    }

    for reactor in PARAMS["reactor_list"]:
        recipes_dict["recipe"].append(reactor.create_recipe())

    return recipes_dict


def region():
    """Return the 'region' part of the Cyclus input file as a dict."""
    d = {
        "region": [
            {"name": "Pakistan", "config": {"NullRegion": None}, **institution()}
        ]
    }
    return d


if __name__ == "__main__":
    main()
