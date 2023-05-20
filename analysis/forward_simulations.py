#!/usr/bin/env python3

"""Scripts to read out results from Cyclus forward simulations."""

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc
from pandas import melt

sns.set_theme(style="darkgrid", context="paper", font="serif", font_scale=1)
rc("axes.formatter", use_mathtext=True)

from sqlite_analyser import AnalyseAllFiles


def main():
    """Main entry point of the script."""
    args = parser()
    analyser = AnalyseAllFiles(
        data_path=args.data_path,
        imgs_path=args.imgs_path,
        job_id=args.job_id,
        max_files=args.max_files,
    )
    analyser.get_data(force_update=args.force_update)

    analyser.plot_1d_histogram("NU_to_reactors")
    analyser.plot_1d_histogram("NU_to_enrichment")
    analyser.plot_1d_histogram("enrichment_feed_NaturalU")
    analyser.plot_1d_histogram("enrichment_feed_SeparatedU")

    # analyser.plot_all_1d_histograms()
    # analyser.pairplots()
    analyser.used_vs_planned_capacity_factor()

    # TODO this plot below may be deleted when uploading stuff in the repository.
    analyser.plot_2d_scatterplots(
        "capacity_factor_planned", "dep_U_mass", marginals=True
    )
    analyser.plot_2d_scatterplots("swu_available", "cs137_mass", marginals=True)
    analyser.plot_2d_scatterplots("swu_available", "dep_U_mass", marginals=True)
    analyser.plot_2d_scatterplots("total_pu", "total_heu", marginals=True)
    analyser.plot_2d_scatterplots(
        "enrichment_feed_SeparatedU", "capacity_factor_planned"
    )
    analyser.plot_2d_scatterplots("enrichment_feed_SeparatedU", "NU_to_enrichment")
    analyser.pairplots(
        subset=[
            "total_pu",
            "total_heu",
            "capacity_factor_planned",
            "capacity_factor_used",
            "swu_used",
            "swu_available",
        ],
        fname="correlations.png",
    )

    print(
        analyser.data[["total_heu", "total_pu"]].describe(
            percentiles=[0.05, 0.1, 0.5, 0.9, 0.95]
        )
    )
    print(args.groundtruth_path, args.groundtruth_jobid)
    final_plots(
        analyser,
        groundtruth_path=args.groundtruth_path,
        groundtruth_jobid=args.groundtruth_jobid,
    )


def final_plots(analyser, groundtruth_path=None, groundtruth_jobid=None):
    g = sns.JointGrid(data=analyser.data, x="total_heu", y="total_pu", height=3)
    g.plot_marginals(sns.histplot, kde=True)
    if groundtruth_path:
        g.plot_joint(sns.scatterplot, label="reconstruction")
        print(
            f"Using groundtruth options with groundtruth_path {groundtruth_path}, groundtruth_jobid {groundtruth_jobid}"
        )
        gt_analyser = AnalyseAllFiles(
            data_path=groundtruth_path,
            imgs_path=Path("imgs", groundtruth_jobid),
            job_id=groundtruth_jobid,
            max_files=10,
        )
        gt_analyser.get_data(force_update=False)
        g.ax_joint.scatter(
            gt_analyser.data["total_pu"],
            gt_analyser.data["total_heu"],
            marker="x",
            color="C3",
            label="groundtruth",
        )
        g.ax_joint.legend()
        g.ax_marg_x.axvline(gt_analyser.data["total_pu"][0], color="C3")
        g.ax_marg_y.axhline(gt_analyser.data["total_heu"][0], color="C3")
    else:
        g.plot_joint(sns.histplot, bins=10)

    g.set_axis_labels("total HEU production [kg]", "total Pu production [kg]")
    g.savefig(analyser.imgs_path / f"scatter_total_pu_total_heu_{analyser.job_id}.pdf")
    plt.close()

    fig, ax = plt.subplots(ncols=2, constrained_layout=True, figsize=(4.5, 2))
    for i, (x, xlabel) in enumerate(
        zip(("total_pu", "total_heu"), ("total Pu [kg]", "total HEU [kg]"))
    ):
        sns.histplot(
            data=analyser.data, x=x, ax=ax[i], kde=False, alpha=0.5, stat="density"
        )
        sns.kdeplot(data=analyser.data, x=x, ax=ax[i], label="reconstruction")
        if groundtruth_path:
            ax[i].axvline(
                gt_analyser.data[x][0],
                linestyle="dashed",
                color="C3",
                label="groundtruth",
            )
        ax[i].set_ylabel("")
        ax[i].set_yticks([])
        ax[i].set_xlabel(xlabel)

    plt.savefig(analyser.imgs_path / f"fissile_material_{analyser.job_id}.pdf")
    plt.close()

    data = analyser.data
    x = "capacity_factor_planned"
    y = "cs137_mass"
    hue = "swu_sampled"
    norm = plt.Normalize(data[hue].min(), data[hue].max())
    cmap = sns.cubehelix_palette(as_cmap=True)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)  # bit hacky to add a colorbar
    sm.set_array([])
    fig, ax = plt.subplots(constrained_layout=True, figsize=(4.5, 2))
    sns.scatterplot(data=analyser.data, x=x, y=y, hue=hue, ax=ax, palette=cmap)
    ax.set_xlabel("capacity factor")
    ax.set_ylabel("Cs137 in waste [kg]")
    ax.get_legend().remove()
    cbar = plt.colorbar(sm, ax=ax, pad=-0.05)
    cbar.ax.set_ylabel("sep. power [kgSWU/year]")
    plt.savefig(analyser.imgs_path / f"scatter_{x}_{y}.pdf")
    plt.close()

    fig, ax = plt.subplots(constrained_layout=True, figsize=(4.5, 2))
    sns.histplot(
        data=melt(
            analyser.data.rename(
                columns={"NU_to_enrichment": "enrichment", "NU_to_reactors": "reactors"}
            ),
            value_vars=("enrichment", "reactors"),
            var_name="consumer",
            value_name="natural uranium consumption [kg]",
        ),
        x="natural uranium consumption [kg]",
        hue="consumer",
        ax=ax,
        kde=True,
        stat="density",
    )
    ax.get_legend().set_title("")
    ax.set_ylabel("")
    ax.set_yticks([])
    plt.savefig(analyser.imgs_path / f"histogram_NU_to_reactor_enrichment.pdf")
    plt.close()

    data = analyser.data
    x = "capacity_factor_planned"
    y = "swu_sampled"
    hue = "enrichment_feed_SeparatedU"
    norm = plt.Normalize(data[hue].min(), data[hue].max())
    cmap = sns.cubehelix_palette(as_cmap=True)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)  # bit hacky to add a colorbar
    sm.set_array([])
    fig, ax = plt.subplots(constrained_layout=True, figsize=(4.5, 2.1))
    sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax, palette=cmap)
    ax.set_xlabel("capacity factor")
    ax.set_ylabel("sep. power [kgSWU/year]")
    ax.get_legend().remove()
    cbar = plt.colorbar(sm, ax=ax, pad=-0.05)
    cbar.ax.set_ylabel("RU enrichment feed [kg]")
    plt.savefig(analyser.imgs_path / f"2dhistogram_params_RU_enrichment_feed.pdf")
    plt.close()

    fig, ax = plt.subplots(constrained_layout=True, figsize=(4.5, 2))
    x = "NU_to_enrichment"
    y = "NU_to_reactors"
    hue = "total_heu"
    norm = plt.Normalize(data[hue].min(), data[hue].max())
    cmap = sns.cubehelix_palette(as_cmap=True)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)  # bit hacky to add a colorbar
    sm.set_array([])
    sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax)
    ax.set_xlabel("NU to enrichment [kg]")
    ax.set_ylabel("NU to reactors [kg]")
    ax.get_legend().remove()
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.ax.set_ylabel("total HEU [kg]")

    plt.savefig(analyser.imgs_path / f"2dhistogram_NU_to_reactor_enrichment.pdf")
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
    parser.add_argument(
        "--force-update",
        action="store_true",
        help=(
            "If set, always extract data from sqlite files. If not set, only "
            "do so in case no presaved data file (data.h5) is available."
        ),
    )
    parser.add_argument("--groundtruth-path", default=None)
    parser.add_argument("--groundtruth-jobid", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
