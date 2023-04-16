#!/usr/bin/env python3

"""Scripts to read out results from Cyclus forward simulations."""

import argparse

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
    # analyser.plot_all_1d_histograms()
    # analyser.pairplots()
    analyser.used_vs_planned_capacity_factor()

    analyser.plot_2d_scatterplots("total_pu", "total_heu", marginals=True)
    analyser.plot_2d_scatterplots(
        "enrichment_feed_SeparatedU", "capacity_factor_planned"
    )
    analyser.plot_2d_scatterplots("enrichment_feed_SeparatedU", "NU_to_enrichment")
    analyser.pairplots(
        subset=["total_pu", "total_heu", "capacity_factor_used", "swu_used"],
        fname="correlations.png",
    )


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
    return parser.parse_args()


if __name__ == "__main__":
    main()
