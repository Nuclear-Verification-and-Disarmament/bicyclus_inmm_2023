#!/usr/bin/env python3

"""Scripts to read out results from Cyclus forward simulations."""

import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpi4py import MPI

sns.set_theme(style="darkgrid")

from sqlite_analyser import SqliteAnalyser


def main():
    """Main entry point of the script."""
    args = parser()
    analyser = AnalyseAllFiles(
        data_path=args.data_path,
        imgs_path=args.imgs_path,
        job_id=args.job_id,
        max_files=args.max_files,
    )
    analyser.get_data()
    analyser.plot_1d_histograms()
    analyser.pairplots()


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
        self.data = None

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
            return

        if os.path.isfile(self.data_fname) and not force_update:
            self.data = pd.read_hdf(self.data_fname)
            print(f"Read in data from {self.data_fname}")
            return

        self.data = []
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
            n_remaining_files = len(self.sqlite_files) % size
            i = 0
            while i < n_remaining_files:
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
        print_mpi(f"Gathered data")

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
            fig, ax = plt.subplots(constrained_layout=True)
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

    def pairplots(self, **hist_kwargs):
        pairplot_grid = sns.PairGrid(self.data, diag_sharey=False)
        pairplot_grid.map_upper(sns.histplot)
        pairplot_grid.map_diag(sns.histplot)
        pairplot_grid.map_lower(sns.scatterplot)
        plt.savefig(self.imgs_path / f"pairplot.png")
        plt.close()


def parser():
    """A simple argparser for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Analyse Cyclus .sqlite output files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        default=".",
        help="Path to output files. The program recursively "
        "walks through all subdirectories.",
    )
    parser.add_argument(
        "--imgs-path", default="imgs/", help="Directory where plots are stored."
    )
    parser.add_argument(
        "--max-files",
        default=None,
        type=int,
        help="If set, do not exceed this amount of files "
        "considered in the analysis.",
    )
    parser.add_argument(
        "--job-id",
        default="",
        type=str,
        help="If set, only consider files with `job-id` in their filename.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
