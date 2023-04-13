#!/usr/bin/env python3

import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpi4py import MPI
from bs4 import BeautifulSoup

sns.set_theme(style="darkgrid")


@dataclass
class AnalyseAllFiles:
    """Class to efficiently analyse a large amount of Cyclus output files."""

    data_path: Union[Path, str]
    imgs_path: Union[Path, str]
    job_id: field(default="")
    max_files: field(default=0)

    def __post_init__(self):
        """Do things immediately after object initialisation."""
        self.data_path = Path(self.data_path)
        self.imgs_path = Path(self.imgs_path)

        self.data_fname = self.imgs_path / "data.h5"
        self.sqlite_files = self.get_files()
        self.data = []

    def get_files(self):
        """Get list of all filenames to be used in the analysis."""
        sqlite_files = []
        for dirpath, _, filenames in os.walk(self.data_path):
            for f in filenames:
                if f.endswith(".sqlite") and self.job_id in f:
                    sqlite_files.append(os.path.join(dirpath, f))

        if self.max_files:
            sqlite_files = sqlite_files[: self.max_files]
        return sqlite_files

    def get_data(self, force_update=False, store_data=True):
        """Extract data from sqlite files using MPI.

        Parameters
        ----------
        force_update : bool, optional
            If True, always extract data from sqlite files.
            If False (default), only do so in case no presaved data file
            (data.h5) is available.

        store_data : bool, optional
            Store data as .h5 file.
        """

        def print_mpi(msg, **kwargs):
            """Helper function to get consistent MPI output."""
            print(
                f"Rank {rank:2}, "
                f"{time.strftime('%y/%m/%d %H:%M:%S', time.localtime())}   " + msg,
                **kwargs,
            )

        if os.path.isfile(self.data_fname) and not force_update:
            self.data = pd.read_hdf(self.data_fname)
            print(f"Read in data from {self.data_fname}")
            return

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            # Maybe not the most elegant solution to distribute tasks but it works
            # and distributes them evenly.
            print_mpi(f"Analysing {len(self.sqlite_files)} in total.")
            n_per_task = len(self.sqlite_files) // size
            files_per_task = [
                self.sqlite_files[i * n_per_task : (i + 1) * n_per_task]
                for i in range(size)
            ]
            i = 0
            while i < len(self.sqlite_files) % size:
                files_per_task[i % size].append(
                    self.sqlite_files[size * n_per_task + i]
                )
                i += 1
        else:
            files_per_task = None

        files_per_task = comm.scatter(files_per_task, root=0)
        print_mpi(f"Received list of {len(files_per_task)} files.")

        # Keys: name of extracted data, values: lists with data
        d = defaultdict(list)
        for i, f in enumerate(files_per_task):
            if i % 50 == 0:
                print_mpi(f"{i:3}/{len(files_per_task)}")

            a = SqliteAnalyser(f, verbose=False)

            d["total_heu"].append(a.material_transfers(-1, "WeapongradeUSink")[1])
            d["total_pu"].append(a.material_transfers(-1, "SeparatedPuSink")[1])
            d["swu_available"].append(a.swu_available("EnrichmentFacility")[1])
            d["cap_factor_planned"].append(a.mean_capacity_factor_planned())

        gatherdata = pd.DataFrame(d)
        gatherdata = comm.gather(gatherdata, root=0)
        print_mpi("Gathered data")

        if rank != 0:
            print_mpi("Exiting function")
            sys.exit()
        print_mpi("Leaving parallelised section.")
        print_mpi("=============================\n")

        print_mpi("Concatenating dataframes")
        data = gatherdata[0]
        for d in gatherdata[1:]:
            data = pd.concat([data, d], axis=0, ignore_index=True)

        self.data = data

        if store_data:
            data.to_hdf(self.data_fname, key="df", mode="w")
            print_mpi(f"Successfully stored data under {self.data_fname}.\n")

    def plot_1d_histograms(self, **hist_kwargs):
        """Generate 1D histograms for all quantities stored in the data.

        Parameters
        ----------
        hist_kwargs : kwargs
            Keyword arguments passed to seaborn.histplot.
        """
        for qty in self.data.columns:
            _, ax = plt.subplots(constrained_layout=True)
            sns.histplot(
                data=self.data,
                x=qty,
                **hist_kwargs,
                label=f"#entries = {len(self.data[qty])}",
                ax=ax,
            )
            ax.legend()
            plt.savefig(self.imgs_path / f"histogram_{qty}.png")
            plt.close()

    def pairplots(self):
        pairplot_grid = sns.PairGrid(self.data, diag_sharey=False)
        pairplot_grid.map_upper(sns.histplot)
        pairplot_grid.map_diag(sns.histplot)
        pairplot_grid.map_lower(sns.scatterplot)
        plt.savefig(self.imgs_path / "pairplot.png")
        plt.close()


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
            print(f"Opened connection to file {self.fname}.")

        self.connection = sqlite3.connect(self.fname)
        self.cursor = self.connection.cursor()
        # self.t_init = self.get_initial_time()
        self.duration = self.cursor.execute("SELECT Duration FROM Info").fetchone()[0]

        # (key,value): (agent_id, name), (name, agent_id),
        #              (spec, agent_id), (agent_id, spec)
        (self.agents, self.names, self.specs, self.agent_ids) = self.get_agents()

    def __del__(self):
        """Destructor closes the connection to the file."""
        try:
            self.connection.close()
            if self.verbose:
                print("Closed connection to file {}.".format(self.fname))
        except AttributeError as e:
            raise RuntimeError(f"Error while closing file {self.fname}") from e

    def get_agents(self):
        """Get all agents that took part in the simulation.

        Returns
        -------
        dict : agent ID (int) -> agent name (str)
        dict : agent name (str) -> agent ID (int)
        dict : agent spec (str) -> agent ID (int)
        dict : agent ID (int) -> agent spec (str)
        """
        query = self.cursor.execute(
            "SELECT Spec, AgentId, Prototype FROM AgentEntry"
        ).fetchall()

        agents = {}
        names = {}
        specs = defaultdict(list)
        agent_ids = {}
        for agent in query:
            # Get the spec (without :agent: or :cycamore: etc. prefix)
            spec = str(agent[0])
            idx = [m.start() for m in re.finditer(":", spec)][-1]
            spec = spec[idx + 1 :]

            agent_id = int(agent[1])
            name = str(agent[2])

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
            except KeyError as e:
                msg = f"Invalid agent name! Valid names are {self.names.keys()}."
                raise KeyError(msg) from e
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
        enter_time = self.cursor.execute(
            "SELECT EnterTime FROM AgentEntry WHERE AgentId = :agent_id",
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
