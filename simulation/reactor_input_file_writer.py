#!/usr/bin/env python3

"""Generate the input file for a Cycamore::Reactor.

This includes an approximated spent fuel composition using a linear model.
"""

import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

import scipy.constants

import pakistan_data
from global_parameters import PARAMS, SECONDS_PER_DAY


@dataclass
class Reactor(ABC):
    """Class to set up the input file of a cycamore::Reactor object.

    Details on some of the parameters passed upon instantiation.

    Parameters
    ----------
    power : float
        In units of thermal MW

    burnup : float
        Target specific burnup, in units MW * d / kg

    capacity_factor : float
        Fraction of online time of the reactor, must be
        0 < capacity_factor < 1.

    fuel_pref : int or list of ints
        Preference used to request fuel in the Cyclus simulation.

    core_mass : float
        In units of kg.

    simulation_ts : int or float
        Duration of one simulation timestep, in units of seconds.
    """

    name: str
    power: float
    burnup: float
    capacity_factor: float
    deployment_year: float
    fuel_pref: Union[int, list]
    core_mass: float
    simulation_ts: float
    decommissioning_year: int
    cycle_time_in_ts: float = field(default=None, init=False)
    refuelling_time_in_ts: float = field(default=None, init=False)
    recipe_name: str = field(init=False)

    def __post_init__(self):
        if self.capacity_factor > 1 or self.capacity_factor <= 0:
            msg = (
                "`capacity_factor` must be <= 1 and > 0. "
                f"`capacity_factor` currently is {capacity_factor}"
            )
            raise ValueError(msg)

        self.recipe_name = Reactor.recipe_name_(self.name)
        if isinstance(self.fuel_pref, (int, float)):
            self.fuel_pref = [self.fuel_pref]
        elif not isinstance(self.fuel_pref, list):
            raise ValueError("'fuel_pref' must be an int or a list of ints.")

        # Burnup is updated within cycle_time_().
        self.cycle_time_()

    def burnup_(self):
        """Calculate the specific burnup."""
        cycle_time_in_days = (
            self.cycle_time_in_ts * self.simulation_ts / SECONDS_PER_DAY
        )
        return self.power * cycle_time_in_days / self.core_mass

    def cycle_time_(self):
        """Calculate the cycle and refuelling times."""
        cycle_time_in_days = self.burnup * self.core_mass / self.power

        # Convert to units of simulation_ts.
        self.cycle_time_in_ts = round(
            cycle_time_in_days * SECONDS_PER_DAY / self.simulation_ts
        )
        self.refuelling_time_in_ts = round(
            self.cycle_time_in_ts * (1.0 - self.capacity_factor) / self.capacity_factor
        )

        # Recalculate burnup because cycle_time may have changed.
        self.burnup = self.burnup_()

    @classmethod
    def recipe_name_(cls, name):
        """Create the spent fuel recipe name of the reactor."""
        return "SpentFuel" + name + "Recipe"

    @abstractmethod
    def create_facility(self):
        """Define the config dict used in the Cyclus input file."""

    @abstractmethod
    def create_recipe(self):
        """Define the spent fuel recipe dict used in the Cyclus input file."""

    def date_to_sim_step(self, date, initial_date):
        """Convert a date (YYYY-MM-DD) into simulation timesteps.

        All dates passed as arguments can be int, float, str or datetimes
        objects.
        - If int or float: must be an integer number, the year, in form YYYY.
        - If str: the date must be in ISO format (YYYY-MM-DD). If day is
          omitted, '01' is assumed. If, additionally, month is omitted, Jan.
          is assumed.

        Parameters
        ----------
        date : int, float, str, datetime.date or datetime.datetime
            Date to be converted

        initial_date : int, float, str, datetime.date or datetime.datetime
            Start day of the simulation, to be subtracted as offset.

        Returns
        -------
        converted_date : int
            Date in units of simulation timesteps.
        """
        if isinstance(date, (int, float)):
            date = f"{date:.0f}"
        if isinstance(date, str):
            date += "-01" * (3 - len(date.split("-")))
            date = datetime.datetime.fromisoformat(date)
        elif isinstance(date, datetime.date):
            date = datetime.datetime(date.year, date.month, date.day)

        if isinstance(initial_date, (int, float)):
            initial_date = f"{initial_date:.0f}"
        if isinstance(initial_date, str):
            initial_date += "-01" * (3 - len(initial_date.split("-")))
            initial_date = datetime.datetime.fromisoformat(initial_date)
        elif isinstance(initial_date, datetime.date):
            initial_date = datetime.datetime(
                initial_date.year, initial_date.month, initial_date.day
            )

        # Convert to time difference in units of seconds using timestamp, then
        # into units of simulation timesteps.
        return round((date.timestamp() - initial_date.timestamp()) / self.simulation_ts)

    def deployment_timestep(self, initial_year):
        """Return the timestep in which the reactor should be deployed.

        Parameters
        ----------
        initial_year : int or float
            Initial year of simulation. Must be smaller or equal to
            `self.deployment_year`.
        """
        if self.deployment_year - initial_year < 0:
            msg = (
                "initial_year must be >= to deployment_year of reactor "
                f"{self.name}. initial_year is {initial_year} and "
                f"deployment_year is {self.deployment_year}"
            )
            raise ValueError(msg)
        return self.date_to_sim_step(self.deployment_year, initial_year)

    def decommissioning_timestep(self, initial_year):
        """Return the lifetime of the reactor.

        Note that this value *generally does not* correspond to the timestep in
        which the reactor gets decommissioned. This correspondance is only true
        if the reactor gets deployed at the beginning of the simulation.
        """
        return self.date_to_sim_step(
            self.decommissioning_year, initial_year
        ) - self.deployment_timestep(initial_year)


class KhushabReactor(Reactor):
    """Class using the Khushab spent fuel composition calculator."""

    def __init__(
        self,
        name,
        power,
        burnup,
        capacity_factor,
        deployment_year,
        fuel_pref,
        core_mass,
        simulation_ts,
        frac_pu=None,
        fuel_incommods=None,
        fuel_inrecipes=None,
        fuel_outcommods=None,
        fuel_outrecipes=None,
        **reactor_kwargs,
    ):
        """Create a Khushab reactor object.

        Parameters
        ----------
        frac_pu : float
            The fraction of Pu239 in the produced plutonium (in units of gPu
            per kgU in spent fuel). If None, then it is calculated using a
            lookup table (see `SpentFuelCalculator.PU_FRAC`).

        reactor_kwargs : kwargs, optional
            All kwargs are passed to the config dictionary, i.e., to
            config['config']['Reactor'].
            **NOTE** Be aware that these kwargs may override any parameters
            set in the dictionary.

        All other parameters : See base class.
        """
        self.frac_pu = frac_pu
        self.reactor_kwargs = reactor_kwargs
        self.fuel_incommods = (
            ["FreshFuel"] if fuel_incommods is None else fuel_incommods
        )
        self.fuel_inrecipes = (
            ["NaturalURecipe"] if fuel_inrecipes is None else fuel_inrecipes
        )
        self.fuel_outcommods = (
            ["SpentFuel"] if fuel_outcommods is None else fuel_outcommods
        )
        self.fuel_outrecipes = fuel_outrecipes

        super().__init__(
            name=name,
            power=power,
            burnup=burnup,
            capacity_factor=capacity_factor,
            deployment_year=deployment_year,
            fuel_pref=fuel_pref,
            core_mass=core_mass,
            simulation_ts=simulation_ts,
            decommissioning_year=-1,
        )

    def create_facility(self, initial_year=None):
        """Create a cycamore::Reactor config dictionary.

        The resulting dictionary can be added directly to the facilities list.
        """
        outrecipes = (
            [self.recipe_name] if self.fuel_outrecipes is None else self.fuel_outrecipes
        )
        config = {
            "name": self.name,
            "config": {
                "Reactor": {
                    "fuel_incommods": {"val": self.fuel_incommods},
                    "fuel_prefs": {"val": self.fuel_pref},
                    "fuel_inrecipes": {"val": self.fuel_inrecipes},
                    "fuel_outcommods": {"val": self.fuel_outcommods},
                    "fuel_outrecipes": {"val": outrecipes},
                    "assem_size": self.core_mass,
                    "n_assem_batch": 1,
                    "n_assem_core": 1,
                    "n_assem_fresh": 0,
                    "cycle_time": self.cycle_time_in_ts,
                    "refuel_time": self.refuelling_time_in_ts,
                    "power_cap": self.power / 3.0,  # Thermal to electric power
                }
            },
        }
        if self.decommissioning_year != -1:
            if initial_year is None:
                msg = (
                    "Reactor {self.name}: You must indicate initial_year, "
                    "else the lifetime cannot be calculated."
                )
                raise ValueError(msg)
            config["lifetime"] = self.decommissioning_timestep(initial_year)

        config["config"]["Reactor"].update(self.reactor_kwargs)

        return config

    def create_recipe(self):
        """Create the recipe dictionary with the spent fuel composition."""
        sfc = SpentFuelCalculator()
        try:
            # No enrichment indicated, defaults to natural uranium.
            sf_comp = sfc.spent_fuel_composition(
                burnup=self.burnup, core_mass=self.core_mass, frac_pu=self.frac_pu
            )
        except Exception as e:
            print(
                f"Reactor {self.name} encountered an error while "
                "the spent fuel composition. Current state of object:\n"
            )
            raise e

        recipe = {
            "name": self.recipe_name,
            "basis": "mass",
            "nuclide": [{"id": iso, "comp": mass} for iso, mass in sf_comp.items()],
        }
        return recipe


class SpentFuelCalculator:
    """Approximate parts of the spent fuel composition of a PHWR.

    The composition of spent fuel from a heavy-water reactor is determined as
    follows:
    - Pu: based on NRX simulations with MCODE (ORIGEN2/MCNP), see
        International Panel on Fissile Materials, "Global Fissile Material
        Report 2010. Balancing the Books: Production and Stocks", Fig B.5,
        upper plot, p. 159. https://fissilematerials.org/library/gfmr10.pdf.
        The plot was digitised and, if needed, a linear interpolation between
        data points is performed. Note that up to a burnup of ~1.43 MWd/kg,
        the Pu is weapongrade (Pu239 fraction > 93%).
    - Cs137: based on 2D Serpent2 simulations of the Savannah River Site
        heavy-water reactor (Mk15 assembly).
    - U235: the number of U235 fissions needed to achieve a given burnup is
        calculated (assuming an energy release of 200 MeV per fission). Hence,
        the mass of U235 in spent fuel is obtained.
    - U238:
    - waste: the products resulting of U235 fission and that are not Cs-137.
        In the output of this program, it is indicated as H1 (hydrogen).
    """

    def __init__(self, burnup=-1):
        """Create a SpentFuelCalculator object. Burnup should be in MWd/kg."""
        self.burnup = burnup  # in thermal MWd/kg.

        self.NRX_CORE_MASS = PARAMS["core_mass"]  # in kg
        self.NRX_INITIAL_ENRICHMENT = 0.00711  # in mass fractions

        # Average recoverable energy per fission of U235 atom.
        self.ENERGY_PER_FISSION = 200e6 * scipy.constants.e  # in J

    @property
    def PU_FRAC(self):
        """Fraction of Pu in spent fuel versus the specific burnup.

        Fraction of Pu in spent fuel (in units of gPu per kgU in spent fuel)
        versus the specific burnup (in thermal MWd/kgHM).

        This variable should not be modified.
        """
        self._PU_FRAC = pakistan_data.pu_df()
        return self._PU_FRAC

    @PU_FRAC.setter
    def PU_FRAC(self):
        """Raises a TypeError because this variable should not be modified."""
        msg = "`SpentFuelCalculator.PU_FRAC` is a variable that should not be changed!"
        raise TypeError(msg)

    @property
    def CS137_FRAC(self):
        """Fraction of Cs137 in spent fuel versus the specific burnup.

        Fraction of Cs137 in spent fuel (in mass fractions) versus the specific
        burnup (in thermal MWd/kgHM).

        This variable should not be modified.
        """
        self._CS137_FRAC = pakistan_data.cs137_df()
        return self._CS137_FRAC

    @CS137_FRAC.setter
    def CS137_FRAC(self):
        """Raises a TypeError because this variable should not be modified."""
        msg = (
            "`SpentFuelCalculator.CS137_FRAC` is a variable that should not"
            " be changed!"
        )
        raise TypeError(msg)

    def linear_interpolation(self, lookup, burnup=None):
        """Perform linear interpolation to get a composition estimate.

        If the burnup value is present in the lookup-table, no interpolation is
        performed. All units are identical to the units used in `lookup`.

        Parameters
        ----------
        lookup : pandas.DataFrame
            The lookup lookup to be used during the interpolation process. It
            must contain two columns, denoted `burnup` and `fraction`.

        burnup : float
            Specific burnup. The unit should be identical to the unit of
            `lookup["burnup"]`, which typically is thermal MWd / kgHM.

        Returns
        -------
        interpolation : float
            Amount of the isotope in question in the spent fuel. The units are
            identical to the units used in `lookup["fraction"]`.

        Raises
        ------
        ValueError
            When `burnup` is smaller/larger than the minimum/maximum burnup in
            `lookup["burnup"]`.
        """
        burnup = self.burnup if burnup is None else burnup

        min_burnup = lookup["burnup"].min()
        max_burnup = lookup["burnup"].max()
        if burnup < min_burnup or burnup > max_burnup:
            msg = (
                f"Selected burnup of {burnup} lies outside the boundaries, "
                f"which are {min_burnup} and {max_burnup}."
            )
            raise ValueError(msg)

        exact_match = lookup[lookup["burnup"] == burnup].index
        if not exact_match.empty:
            return lookup["fraction"][exact_match].iloc[0]

        # Get indices of neighbours.
        lower_nb_idx = lookup[lookup["burnup"] < burnup]["burnup"].idxmax()
        upper_nb_idx = lookup[lookup["burnup"] > burnup]["burnup"].idxmin()

        # Perform linear interpolation
        dy = lookup["fraction"][upper_nb_idx] - lookup["fraction"][lower_nb_idx]
        dx = lookup["burnup"][upper_nb_idx] - lookup["burnup"][lower_nb_idx]
        x = burnup - lookup["burnup"][lower_nb_idx]
        interpolation = dy / dx * x + lookup["fraction"][lower_nb_idx]

        return interpolation

    def spent_fuel_composition(
        self, burnup=None, core_mass=None, frac_pu=None, initial_enrichment=None
    ):
        """Calculate parts of the spent fuel composition.

        Parameters
        ----------
        burnup : float, optional
            The specific burnup in units of thermal MWd / kgHM.

        core_mass : float, optional
            The initial mass of the full core in kg.

        frac_pu : float, optional
            The fraction of Pu239 in the produced plutonium (in units of gPu
            per kgU in spent fuel). If None, then it is calculated using a
            lookup table (see `SpentFuelCalculator.PU_FRAC`).

        initial_enrichment : float, optional
            The initial enrichment grade of the uranium in mass fractions.

        Returns
        -------
        spent fuel composition : dict
            A dictionary containing the isotope masses in the spent fuel in kg.
            The keys denote the isotopes in the format "Cs137", "U235M", etc.
            This allows easy integration into the Cyclus input file-script.
            The 'waste' is indicated as H1.
        """
        burnup = self.burnup if burnup is None else burnup
        core_mass = self.NRX_CORE_MASS if core_mass is None else core_mass
        initial_enrichment = (
            self.NRX_INITIAL_ENRICHMENT
            if initial_enrichment is None
            else initial_enrichment
        )

        MOLAR_U235 = 235e-3  # molar mass of U235 in kg/mol
        # Number and mass of U235 atoms fissioned to get to the burnup. This
        # calculation neglects the Pu239 fissions, which is fine, though, for
        # the low burnups considered here. The factor 1e6 is the conversion from MJ to J.
        # The core mass is needed because the specific burnup is indicated.
        n_u235_fissioned = (
            burnup * SECONDS_PER_DAY * 1e6 * core_mass / self.ENERGY_PER_FISSION
        )
        m_u235_fissioned = n_u235_fissioned * MOLAR_U235 / scipy.constants.N_A
        m_u235_final = initial_enrichment * core_mass - m_u235_fissioned

        # The fraction of Pu is taken with respect to the mass of uranium in
        # *spent* fuel (as opposed to the initial core mass). I assume that
        # all fissioned U235 fissions into smaller nuclei (no neutron capture).
        if frac_pu is None:
            frac_pu = self.linear_interpolation(self.PU_FRAC, burnup=burnup)

        m_pu = 1e-3 * frac_pu * (core_mass - m_u235_fissioned)  # in kg

        # Calculate Cs-137 mass in spent fuel.
        frac_cs137 = self.linear_interpolation(self.CS137_FRAC, burnup=burnup)
        m_cs137 = frac_cs137 * core_mass  # in kg

        # Nuclear waste, i.e., products from U235 fission excluding Cs-137.
        m_waste = m_u235_fissioned - m_cs137

        # Calculate U-238 mass in spent fuel.
        m_u238_final = core_mass * (1 - initial_enrichment) - m_pu  # in kg

        return {
            "U235": m_u235_final,
            "U238": m_u238_final,
            "Cs137": m_cs137,
            "Pu239": m_pu,
            "H1": m_waste,
        }
