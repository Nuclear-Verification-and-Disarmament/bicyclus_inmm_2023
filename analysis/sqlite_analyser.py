#!/usr/bin/env python3

import os
import re
import sqlite3
from collections import defaultdict

import numpy as np
from bs4 import BeautifulSoup


class SqliteAnalyser:
    """Class to analyse Cyclus .sqlite output files."""

    def __init__(self, fname, verbose=True):
        """Initialise Analyser object.

        Parameters
        ----------
        fname : Path
            Path to .sqlite file
        """
        self.fname = fname
        if not os.path.isfile(self.fname):
            msg = f"File {os.path.abspath(self.fname)} does not exist!"
            raise FileNotFoundError(msg)

        self.verbose = verbose
        if self.verbose:
            print("Opened connection to file {}.".format(self.fname))

        self.connection = sqlite3.connect(self.fname)
        self.cursor = self.connection.cursor()
        # self.t_init = self.get_initial_time()
        self.duration = self.cursor.execute("SELECT Duration FROM Info").fetchone()[0]

        # (key,value): (agent_id, name), (name, agent_id),
        #              (spec, agent_id), (agent_id, spec)
        (self.agents, self.names, self.specs, self.agent_ids) = self.get_agents()

        return

    def __del__(self):
        """Destructor closes the connection to the file."""
        try:
            self.connection.close()
            if self.verbose:
                print("Closed connection to file {}.".format(self.fname))
        except AttributeError as e:
            raise RuntimeError("Error while closing file {self.fname}") from e

    def get_agents(self):
        """Get all agents that took part in the simulation.

        Returns
        -------
        dict : agent ID (int) -> agent name (str)
        dict : agent name (str) -> agent ID (int)
        dict : agent spec (str) -> agent ID (int)
        dict : agent ID (int) -> agent spec (str)
        """
        data = self.cursor.execute(
            "SELECT Spec, AgentId, Prototype FROM AgentEntry"
        ).fetchall()

        agents = {}
        names = {}
        specs = defaultdict(list)
        agent_ids = {}
        for i in range(len(data)):
            # Get the spec (without :agent: or :cycamore: etc. prefix)
            spec = str(data[i][0])
            idx = [m.start() for m in re.finditer(":", spec)][-1]
            spec = spec[idx + 1 :]

            agent_id = int(data[i][1])
            name = str(data[i][2])

            agents[agent_id] = name
            names[name] = agent_id
            specs[spec].append(agent_id)
            agent_ids[agent_id] = spec

        return agents, names, specs, agent_ids

    def agent_id_or_name(self, agent_id_or_name):
        """Helper function to convert agent names into agent IDs."""
        if isinstance(agent_id_or_name, str):
            try:
                return self.names[agent_id_or_name]
            except KeyError:
                msg = f"Invalid agent name! Valid names are {self.names.keys()}."
                raise KeyError(msg)
        elif isinstance(agent_id_or_name, int):
            return agent_id_or_name
        else:
            raise ValueError("`agent_id_or_name` must be agent name (str) or id (int)!")

    def material_transfers(self, sender_id_or_name, receiver_id_or_name, sum_=True):
        """Get all material transfers between two facilities.

        Parameters
        ----------
        sender_id_or_name, receiver_id_or_name : int or str
            Agent IDs or agent names of sender and receiver, respectively. Use
            '-1' as placeholder for 'all facilities'.

        sum_ : bool, optional
            If true, sum over all timesteps.

        Returns
        -------
        transfer_array : np.array of shape (number of transfers, 2)
            The first element of each entry is the timestep of the transfer,
            the second is its mass. If sum_ is True, then the second element
            is the total mass (over all timesteps) and the time is set to -1.
        """
        sender_id = self.agent_id_or_name(sender_id_or_name)
        receiver_id = self.agent_id_or_name(receiver_id_or_name)

        sender_cond = "" if sender_id == -1 else "SenderId = :sender_id "
        recv_cond = "" if receiver_id == -1 else "ReceiverId = :receiver_id "

        if sender_cond and recv_cond:
            sqlite_condition = f"WHERE ({sender_cond} AND {recv_cond})"
        else:
            sqlite_condition = f"WHERE ({sender_cond}{recv_cond})"

        sender_receiver = {"sender_id": sender_id, "receiver_id": receiver_id}
        transfer_times = self.cursor.execute(
            f"SELECT Time FROM Transactions {sqlite_condition}", sender_receiver
        ).fetchall()
        transfer_masses = self.cursor.execute(
            "SELECT Quantity FROM Resources WHERE ResourceId IN "
            f"(SELECT ResourceId FROM Transactions {sqlite_condition});",
            sender_receiver,
        ).fetchall()

        transfer_array = np.array(
            [[time[0], mass[0]] for time, mass in zip(transfer_times, transfer_masses)]
        )

        if sum_:
            return np.array([-1, transfer_array[:, 1].sum()])

        return transfer_array

    def swu_available(self, agent_id_or_name, sum_=True):
        """Get the SWU available to one enrichment facility.

        TODO COMPLETE DOCSTRING

        Parameters
        ----------

        TODO COMPLETE DOCSTRING
        Returns
        -------
        TODO COMPLETE DOCSTRING
        """
        agent_id = self.agent_id_or_name(agent_id_or_name)
        enter_time, lifetime = self.cursor.execute(
            "SELECT EnterTime, Lifetime FROM AgentEntry WHERE AgentId = :agent_id",
            {"agent_id": agent_id},
        ).fetchone()
        data = self.cursor.execute(
            "SELECT swu_capacity_times, swu_capacity_vals FROM "
            "AgentState_flexicamore_FlexibleEnrichmentInfo WHERE AgentId = :agent_id",
            {"agent_id": agent_id},
        ).fetchone()  # (Boost vector with times, Boost vector with SWUs)

        # Convert Boost vectors (in XML) into Python lists.
        swu_times = []
        swu_vals = []
        for list_, cyclus_data in zip(
            (swu_times, swu_vals), [BeautifulSoup(d, "xml") for d in data]
        ):
            for item_ in cyclus_data.find_all("item"):
                list_.append(float(item_.get_text()))

        # Fill with timesteps where SWU was not changed.
        complete_list = []
        previous_time = swu_times[0]
        previous_val = swu_vals[0]
        for time, val in zip(swu_times, swu_vals):
            for t in range(int(previous_time), int(time)):
                complete_list += [[t, previous_val]]
            previous_time = time
            previous_val = val

        # Convert to array and transform timesteps since deployment to timesteps since
        # start of the simulation.
        swu_available = np.array(complete_list)
        swu_available[:, 0] += enter_time

        if sum_:
            return np.array([-1, swu_available[:, 1].sum()])
        return swu_available

    def capacity_factor_planned(self, agent_id_or_name):
        """Get the planned capacity factor of one reactor.

        Note that this value corresponds to the capacity factor *as indicated
        in Cyclus' input file. Thus, the actual capacity factor (online time /
        total time) may be smaller than this value, e.g., in case of missing
        fresh fuel.

        Parameters
        ----------
        agent_id_or_name : str or int
            Agent ID or agent name
        """
        agent_id = self.agent_id_or_name(agent_id_or_name)
        if self.agent_ids[agent_id] != "Reactor":
            msg = f"Agent ID {agent_id} does not correspond to a 'Reactor' facility"
            raise ValueError(msg)

        cycle_time, refuelling_time = self.cursor.execute(
            "SELECT cycle_time, refuel_time FROM AgentState_cycamore_ReactorInfo "
            "WHERE AgentId = :agent_id",
            {"agent_id": agent_id},
        ).fetchone()
        return cycle_time / (cycle_time + refuelling_time)

    def mean_capacity_factor_planned(self, spec="Reactor"):
        """Get the mean planned capacity factor of all reactors.

        Note that this is the arithmetic mean *weighted* with each reactor's
        lifetime.
        """
        agent_ids = [id_ for id_, spec_ in self.agent_ids.items() if spec_ == spec]
        reactor_specs = {}
        for agent_id in agent_ids:
            enter_time, lifetime = self.cursor.execute(
                "SELECT EnterTime, Lifetime FROM AgentEntry WHERE AgentId = :agent_id",
                {"agent_id": agent_id},
            ).fetchone()
            in_sim_time = lifetime if lifetime != -1 else self.duration - enter_time
            reactor_specs[agent_id] = {
                "in_sim_time": in_sim_time,
                "capacity_factor": self.capacity_factor_planned(agent_id),
            }

        numerator = sum(
            r["in_sim_time"] * r["capacity_factor"] for r in reactor_specs.values()
        )
        denominator = sum(r["in_sim_time"] for r in reactor_specs.values())
        return numerator / denominator
