"""A collection of functions useful to generate input files."""

from datetime import date, datetime

from pandas import Series

from global_parameters import PARAMS, SECONDS_PER_YEAR


def yearly_rates_to_sim_steps(var, time):
    """Convert yearly rates in `var` into simulation timestep rates.

    Parameters
    ----------
    var : array-like
        Array to be converted. Must contain *yearly* rates.
    time : array-like
        Corresponding years.

    Returns
    -------
    converted_var : list
        Has as many entries as `var` but in units of simulation timestep rates.
    converted_time : list
        Years from `time` converted to simulation timesteps.
    """
    if len(var) != len(time):
        raise RuntimeError("'var' and 'time' must have the same dimension.")

    if isinstance(time, Series):
        t0 = time.iloc[0]
    else:
        t0 = time[0]

    sim_steps_per_year = SECONDS_PER_YEAR / PARAMS["dt"]
    converted_var = [v / sim_steps_per_year for v in var]
    converted_time = [round((t - t0) * sim_steps_per_year) for t in time]

    return converted_var, converted_time


def date_to_sim_step(date):
    """Convert a date (YYYY-MM-DD) into simulation timesteps.

    Parameters
    ----------
    date : int, float, str, datetime.date or datetime.datetime
        Can be a str or a datetime object.
        If int or float: must be an integer number, the year, in the form YYYY.
        If str: the date must be in ISO format (YYYY-MM-DD). If day is omitted,
        '01' is assumed. If, additionally, month is omitted, Jan. is assumed.

    Returns
    -------
    converted_date : int
        Date in units of simulation timesteps.
    """
    if isinstance(date, (int, float)):
        date = f"{date:.0f}"
    if isinstance(date, str):
        date += "-01" * (3 - len(date.split("-")))
        date = datetime.fromisoformat(date)
    elif isinstance(date, date):
        date = datetime(date.year, date.month, date.day)

    # Convert to time difference in units of seconds using timestamp, then into
    # units of simulation timesteps.
    return round((date.timestamp() - PARAMS["startdate"].timestamp()) / PARAMS["dt"])


def add_reprocessing(
    name,
    commod,
    separation_eff,
):
    """Generate the config dictionary for a reprocessing facility.

    Params
    ------
    name : str
    commod : str
    separation_eff : float
    """
    config = {
        "name": name,
        "config": {
            "Separations": {
                "feed_commods": {"val": [commod]},
                "feedbuf_size": 1e299,
                "leftover_commod": "SeparatedWaste",
                "streams": {
                    "item": [
                        {
                            "commod": "SeparatedPu",
                            "info": {
                                "buf_size": 1e299,
                                "efficiencies": {
                                    "item": [{"comp": "94000", "eff": separation_eff}]
                                },
                            },
                        },
                        {
                            "commod": "SeparatedU",
                            "info": {
                                "buf_size": 1e299,
                                "efficiencies": {
                                    "item": [
                                        {"comp": "92235", "eff": separation_eff},
                                        {"comp": "92238", "eff": separation_eff},
                                    ]
                                },
                            },
                        },
                    ]
                },
            }
        },
    }
    return config


def add_sink(commod, name=None, recipe=None):
    """Generate the Sink configuration dictionary.

    Parameters
    ----------
    commod : str or list of str

    name : str, optional
        If omitted, the facility will be called `commod`Sink.

    recipe : str, optional
        If None, default mode is used (recipe is called `commod`Recipe).
        If set to '', then the recipe entry is omitted in the config file.
    """
    commod = [commod] if isinstance(commod, str) else commod
    name = name if name is not None else commod[0] + "Sink"
    recipe = recipe if recipe is not None else commod[0] + "Recipe"
    config = {
        "name": name,
        "config": {
            "Sink": {
                "in_commods": {"val": commod},
            }
        },
    }
    if recipe:
        config["config"]["Sink"]["recipe_name"] = recipe

    return config


def add_source(name, commod, recipe=None, source_kwargs=None, facility_kwargs=None):
    """Return a Source config dictionary.

    Parameters
    ----------
    source_kwargs : optional
        Source state variables. If omitted, Cycamore defaults are used.
    facility_kwargs : optional
        Facility state variables. If omitted, Cycamore defaults are used.
    """
    source_kwargs = {} if source_kwargs is None else source_kwargs
    facility_kwargs = {} if facility_kwargs is None else facility_kwargs
    recipe = recipe if recipe is not None else commod + "Recipe"
    config = {
        "name": name,
        "config": {"Source": {"outcommod": commod, **source_kwargs}},
        **facility_kwargs,
    }
    if recipe:
        config["config"]["Source"]["outrecipe"] = recipe
    return config


def add_storage(name, in_commod, out_commod, recipe=None, **storage_kwargs):
    """Return a storage config dictionary.

    Parameters
    ----------
    in_commod, out_commod : str
        Respective commodities

    recipe : str, optional
        If None, then `in_commod + 'Recipe'` is used. If you do not want to
        indicate a recipe, use ''.

    storage_kwargs : optional
        Storage state variables. If omitted, Cycamore defaults are used.
    """
    recipe = recipe if recipe is not None else in_commod + "Recipe"
    config = {
        "name": name,
        "config": {
            "Storage": {
                "in_commods": {"val": [in_commod]},
                "out_commods": {"val": [out_commod]},
                **storage_kwargs,
            }
        },
    }
    if recipe:
        config["config"]["Storage"]["in_recipe"] = recipe
    return config


def add_uranium_recipe(name, basis, u235):
    """Return a Cyclus recipe for a U235-U238 mixture.

    Parameters
    ----------
    name : str
        Recipe name
    basis : str
        Whether to use atom or mass fractions. Must be 'atom' or 'mass.
    u235 : float
        Fraction of U235 in the mixture. Must be 0 < u235 < 1.
    """
    if u235 <= 0 or u235 > 1:
        raise ValueError("'u235' must be > 0 and < 1.")
    if basis not in ("atom", "mass"):
        raise ValueError("'basis' must be 'atom' or 'mass'.")

    return {
        "name": name,
        "basis": basis,
        "nuclide": [{"id": "U235", "comp": u235}, {"id": "U238", "comp": 1 - u235}],
    }
