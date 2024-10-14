#!/usr/bin/env python3

import os
import logging
import tempfile

from perf_llm_eval.parser import setup_parser

logger = logging.getLogger("perf-llm-eval")


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

    # Setup environment
    ## Disable downloads from hf
    os.environ["HF_HUB_OFFLINE"] = "1"
    ## Unitxt need to set this to run certain benchmarks
    os.environ["UNITXT_ALLOW_UNVERIFIED_CODE"] = "True"
    ## If we don't set this then Unitxt
    ## will redownload the catalog each run
    os.environ["UNITXT_USE_ONLY_LOCAL_CATALOGS"] = "True"
    ## Use local Unitxt catalog
    os.environ["UNITXT_ARTIFACTORIES"] = args.catalog_path

    # Late import to avoid slow cli
    from perf_llm_eval.lm_eval_wrapper import LMEvalWrapper

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
            os.symlink(f"{local_dir}/unitxt", f"{tmpdir}/unitxt")

            # Symlink datasets to working directory
            for dataset in os.listdir(args.datasets):
                if dataset == 'unitxt':
                    continue # TODO Handle this better
                os.symlink(f"{args.datasets}/{dataset}", f"{tmpdir}/{dataset}")

            # Call wrapped lm-eval
            args.tasks = args.tasks.split(',')
            LMEvalWrapper.exec(**vars(args))


if __name__ == '__main__':
    eval_cli()
