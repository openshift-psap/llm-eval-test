#!/usr/bin/env python3

import os
import logging
import tempfile
from typing import Optional

from llm_eval_test.parser import setup_parser

logger = logging.getLogger("llm-eval-test")


def config_env(offline_mode: bool = True, unitxt_catalog: Optional[str] = None):
    """Setup environment."""

    # Unitxt need to set this to run certain benchmarks
    os.environ["UNITXT_ALLOW_UNVERIFIED_CODE"] = "True"

    if offline_mode:
        # Disable downloads from hf
        os.environ["HF_HUB_OFFLINE"] = "1"
        # If we don't set this then Unitxt
        # will redownload the catalog each run
        os.environ["UNITXT_USE_ONLY_LOCAL_CATALOGS"] = "True"

    if unitxt_catalog:
        # Use local Unitxt catalog
        # NOTE: If UNITXT_USE_ONLY_LOCAL_CATALOGS is not set the
        # default catalog cards may overwrite local cards
        os.environ["UNITXT_ARTIFACTORIES"] = unitxt_catalog


def eval_cli():
    local_dir = os.path.dirname(__file__)
    work_dir = os.getcwd()

    parser = setup_parser(local_dir, work_dir)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(levelname)-.4s %(asctime)s,%(msecs)03d [%(name)s@%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=args.loglevel
    )
    logger.info("CLI called with " + str(vars(args)))

    config_env(unitxt_catalog=args.catalog_path)

    # Late import to avoid slow cli
    from llm_eval_test.lm_eval_wrapper import LMEvalWrapper

    if args.command == 'list':
        LMEvalWrapper.list_tasks(args.tasks_path)
    elif args.command == 'run':
        # HACK: Working from a temporary directory allows us to load hf datasets
        # from disk because the dataset and evaluate libraries search the local
        # path first. Since Unitxt is loaded as a dataset, we also provide wrappers
        # that point to the local python package.
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            logger.info(f"Changing working directory to {tmpdir}")

            # Symlink unitxt wrappers to working directory
            for dataset in os.listdir(f"{local_dir}/wrappers"):
                os.symlink(f"{local_dir}/wrappers/{dataset}", f"{tmpdir}/{dataset}")

            # Symlink datasets to working directory
            for dataset in os.listdir(args.datasets):
                try:
                    os.symlink(f"{args.datasets}/{dataset}", f"{tmpdir}/{dataset}")
                except FileExistsError:
                    logger.warn(f"Dataset '{dataset}' conflicts with existing wrapper, skipping")

            # Call wrapped lm-eval
            args.tasks = args.tasks.split(',')
            LMEvalWrapper.exec(**vars(args))


if __name__ == '__main__':
    eval_cli()
