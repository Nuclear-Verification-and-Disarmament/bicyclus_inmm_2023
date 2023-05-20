#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid", context="paper", font="serif", font_scale=1)


def main():
    get_posterior()


def get_posterior():
    job_ids = (34397340, 34397257)
    add_groundtruth = True
    if add_groundtruth:
        with open(
            Path("../simulation/parameters/reconstruction/swu_cap_factor_true.json"),
            "r",
        ) as f:
            groundtruth = json.load(f)

    for job_id in job_ids:
        path = Path(f"imgs/{job_id}/{job_id}_{job_id}.cdf")
        data = az.from_netcdf(path)

        _, ax = plt.subplots(ncols=2, constrained_layout=True, figsize=(4.5, 2))
        for axis, xlabel in zip(ax, ("capacity factor", "separative power [kgSWU/yr]")):
            axis.set_title("")
            axis.set_xlabel(xlabel)
        for axis, xlabel in zip(ax, ("global_capacity_factor", "swu_increase2")):
            sns.histplot(
                data=data["posterior"][xlabel].values.flatten(), ax=axis, kde=True
            )
            axis.set_ylabel("")
            axis.set_yticks([])
            axis.axvline(groundtruth[xlabel], linestyle="dashed", color="C3")

        fname = path.parent / f"posterior_{job_id}.pdf"
        print(f"Saving plot under {fname}")
        plt.savefig(fname)
        plt.close()

        var1, var2 = data.posterior
        lo, hi = 0, 1
        x, y = np.array(data.posterior[var1]), np.array(data.posterior[var2])
        x = x.reshape(x.size)
        y = y.reshape(y.size)
        mask = (np.isnan(x)) | (np.isnan(y))
        x = x[~mask]
        y = y[~mask]

        xrange = (np.quantile(x, lo), np.quantile(x, hi))
        yrange = (np.quantile(y, lo), np.quantile(y, hi))

        df = data.to_dataframe().rename(
            columns={("posterior", v): v for v in (var1, var2)}
        )

        _, ax = plt.subplots(constrained_layout=True, figsize=(4, 3))
        sns.kdeplot(data=df, x=var1, y=var2, fill=True, ax=ax)
        ax.set_xlim(0.6, 0.80)
        ax.set_ylim(20e3, 45e3)
        ax.set_xlabel("capacity factor")
        ax.set_ylabel("separative power [kgSWU/yr]")
        fname = path.parent / f"plot_merge_histogram_{job_id}.pdf"
        print(f"Saving plot under {fname}")
        plt.savefig(fname)
        plt.close()


if __name__ == "__main__":
    main()
