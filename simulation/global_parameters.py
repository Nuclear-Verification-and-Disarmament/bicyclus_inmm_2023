#!/usr/bin/env python3

""""A script with constants and parameters used in the Cyclus input file."""


PARAMS = {
    # Control parameters
    # ==================
    "decay": "lazy",
    "duration": None,  # duration is *not* sampled, it must be initialised to
    # `None`, else  other functions (e.g., `yearly_rates_to_sim_steps`) will not
    # work as expected. This value will be updated, see `control()`. Also see
    # `GLOBAL_PARAMS['dt'] for the unit.
    "dt": 86400,  # duration of one simulation timestep in seconds
    # 86400 s = 1 day, 432000 s = 5 days
    "endyear": 2023,  # Exclusive, i.e., the simulation ends on Dec., 31 of
    # the previous year.
    "explicit_inventory": False,
    "explicit_inventory_compact": False,
    "simhandle": "Military NFC",  # Identifier for simulation
    "solver": "greedy",  # Use `greedy` or `coin-or` or `coinor`.
    "startmonth": 1,  # January
    "startyear": 1983,
    "startdate": None,  # Will be set upon execution.
    # Facility parameters
    # ===================
    # Storage parameters
    # ------------------
    "add_waste_sink_in_last_step": True,  # Force a decay calculation at the end
    # of the simulation. DO NOT SET THIS TO `FALSE` when using in
    # conjunction with PyMC, else the sampling process may fail.
    # Enrichment parameters
    # ---------------------
    # Additional information on the SWU variables: see pakistan_data/swu_df.
    "swu_increase2": 30e3,  # Must be >= 15000. In kgSWU/year
    "swu_increase3": None,  # Either None or > swu_increase2. In kgSWU/year
    "max_swu_per_day": 100000 / 365,  # Value hardcoded in kg/day
    "enrichment_max_feed_inv": 400,  # Value is *hardcoded* in kg/day. It
    # corresponds roughly to the feed used for the enrichment of 0.6% U to
    # 93% with 0.3% tails and 274 kgSWU/day (~100'000 kgSWU/year).
    # Khushab parameters
    # ------------------
    "core_mass": 9450.0,  # in kg, mass of an NRX reactor core
    "global_capacity_factor": None,  # Warning: if not None, then this will
    # override all 4 individual factors.
    "khushab1_burnup": 1.2,  # in MWth d / kg. Point at which Pu goes from
    "khushab2_burnup": 1.2,  # weapongrade to reactorgrade, following
    "khushab3_burnup": 1.2,  # GFMR 2010 p. 159.
    "khushab4_burnup": 1.2,
    "khushab1_capacity_factor": 0.7,  # in fractions
    "khushab2_capacity_factor": 0.7,
    "khushab3_capacity_factor": 0.7,
    "khushab4_capacity_factor": 0.7,
    "khushab_frac_pu": 0.74,  # Must be in units of gPu / kgU or None.
    # If None, calculate fraction of Pu239 in spent
    # fuel from lookup table.
    "khushab1_power": 49.0,  # in MWth, following T. Patton, SGS 20:137 (2012)
    "khushab2_power": 66.0,  # following T. Patton, SGS 20:137 (2012)
    "khushab3_power": 81.0,  # following T. Patton, SGS 20:137 (2012)
    "khushab4_power": 100.0,  # very rough estimate.
    # Deploment years. For simplicity assume all reactors to be deployed on
    # January, 1st of that year. References:
    "khushab1_deployment": 1998,  # Mian, SGS17 (2009); Albright, ISIS report (1998)
    "khushab2_deployment": 2010,  # Mian, IPFM Blog (2010); Brannan, ISIS report (2010)
    "khushab3_deployment": 2013,  # IPFM Blog June 2014
    "khushab4_deployment": 2015,  # Albright and Kelleher-Vergantini, ISIS report (2015)
    "khushab_fuel_pref": 30,  # At the moment, use one global preference.
    "reactor_list": [],  # Will be updated in `cyclus_input.py`.
    # Separations/reprocessing parameters
    # -----------------------------------
    "separation_eff": 0.99,  # Following Wilson `The Nuclear Fuel Cycle` (1996)
    # Source/mining parameters
    # ------------------------
    "initial_u_stockpile": 339e3,  # Initial NU stockpile in kg
    "current_mining_production": 45000,  # Current yearly NU production in kg
    # Recipe parameters (also partly used for enrichment stuff)
    # =========================================================
    "NU_assay_mass": 0.00711,  # in mass fractions
    "DU_enrichment_atom": 0.003,  # in atom fractions
    "WGU_enrichment_atom": 0.90,  # in atom fractions
}
SECONDS_PER_DAY = 86400
SECONDS_PER_YEAR = SECONDS_PER_DAY * 365
