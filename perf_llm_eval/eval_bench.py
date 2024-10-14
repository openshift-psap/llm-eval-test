#!/usr/bin/env python3

import os
import argparse
import logging
import tempfile

logger = logging.getLogger("perf-llm-eval")


def setup_parser(local_dir: str, work_dir: str):
    def dir_path(path: str) -> str:
        ''' Typecheck for directory '''
        if os.path.isdir(path):
            return os.path.abspath(path)
        else:
            raise NotADirectoryError(path)

    parser_base = argparse.ArgumentParser(add_help=False)
    parser_base.add_argument('--catalog_path', type=dir_path,
                        default=f"{local_dir}/lm_eval/catalog",
                        help="unitxt catalog directory")
    parser_base.add_argument('--tasks_path', type=dir_path,
                        default=f"{local_dir}/lm_eval/tasks",
                        help="lm-eval tasks directory")
    log_group = parser_base.add_mutually_exclusive_group()
    log_group.add_argument('--verbose', '-v', default=logging.INFO,
                           action="store_const", dest="loglevel", const=logging.DEBUG,
                           help="set loglevel to DEBUG")
    log_group.add_argument('--quiet', '-q',
                           action="store_const", dest="loglevel", const=logging.ERROR,
                           help="set loglevel to ERROR")


    parser = argparse.ArgumentParser(
        description='Helper script for running PSAP relevent LLM benchmarks.',
        epilog='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parser_base]
    )
    subparsers = parser.add_subparsers(help='commands', dest='command')
    parser_run = subparsers.add_parser(
        'run',
        description="Run tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parser_base]
    )
    parser_run.add_argument('--endpoint', '-H', required=True,
                        default='http://127.0.0.1:8000/v1/completions',
                        help='OpenAI API-compatible endpoint')
    parser_run.add_argument('--model', '-m', required=True,
                        help="name of the model under test")
    parser_run.add_argument('--tasks', '-t', required=True,
                        help="comma separated list of tasks")
    parser_run.add_argument('--datasets', '-d', type=dir_path, required=True,
                        default=f"{work_dir}/datasets",
                        help="path to dataset storage")
    parser_run.add_argument('--batch_size', '-b', default=64, type=int,
                        help="per-request batch size")
    parser_run.add_argument('--output', '-o', type=argparse.FileType('w'),
                        default=f"{work_dir}/output.json",
                        help="results output file")

    parser_list = subparsers.add_parser(
        'list',
        description="List available tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parser_base]
    )

    return parser


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
