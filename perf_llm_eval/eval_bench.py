#!/usr/bin/env python3

import os
import argparse
import logging
import tempfile
import json
import dotenv

logger = logging.getLogger("perf-llm-eval")

dotenv.load_dotenv()

def exec_lm_eval(tasks, model, endpoint, **kwargs):

    # Avoid importing these until we want to exec
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.tasks import TaskManager  # type: ignore
    from lm_eval.utils import make_table

    model_args = dict(
        model = model,
        base_url = endpoint,
        num_concurent=1,
        max_retries=3,
        tokenizer_backend=None,
        tokenized_requests=False
    )

    model_args_str = ','.join([f"{k}={str(v)}" for k,v in model_args.items()])
    tm = TaskManager(
        include_path=kwargs['tasks_path'],
        include_defaults=False,
        verbosity=logging.getLevelName(kwargs['loglevel'])
    )

    logger.info("Running lm-eval")
    results = simple_evaluate(
        model="local-completions",
        model_args=model_args_str,
        tasks=tasks,
        #num_fewshot=self.few_shots,
        batch_size=kwargs['batch_size'],
        task_manager=tm,
        verbosity=logging.getLevelName(kwargs['loglevel'])
    )

    if results:
        # Write results to outfile
        logger.info(f"Writing results to {kwargs['output'].name}")
        output = json.dumps(results, indent=2, ensure_ascii=False)
        kwargs['output'].write(output)

        # Print output table
        print(make_table(results))
        if results.get('groups'):
            print(make_table(results, "groups"))

def eval_cli():
    parser = argparse.ArgumentParser(
                    description='Helper script for running PSAP relevent LLM benchmarks.',
                    epilog='',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    local_dir = os.path.dirname(__file__)
    work_dir = os.getcwd()

    def dir_path(path: str) -> str:
        ''' Typecheck for directory '''
        if os.path.isdir(path):
            return os.path.abspath(path)
        else:
            raise NotADirectoryError(path)

    parser.add_argument('--catalog_path', type=dir_path,
                        default=f"{local_dir}/lm_eval/catalog",
                        help="unitxt catalog directory")
    parser.add_argument('--tasks_path', type=dir_path,
                        default=f"{local_dir}/lm_eval/tasks",
                        help="unitxt catalog directory")
    parser.add_argument('--datasets', '-d', type=dir_path, required=True,
                        default=f"{work_dir}/datasets",
                        help="path to dataset storage")
    parser.add_argument('--endpoint', '-H', required=True,
                        default='http://127.0.0.1:8000/v1/completions',
                        help='OpenAI API-compatible endpoint')
    parser.add_argument('--model', '-m', required=True,
                        help="name of the model under test")
    parser.add_argument('--batch_size', '-b', default=64, type=int,
                        help="per-request batch size")
    parser.add_argument('--tasks', '-t', required=True,
                        help="comma separated list of tasks")
    parser.add_argument('--output', '-o', type=argparse.FileType('w'),
                        default=f"{work_dir}/output.json",
                        help="results output file")
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument('--verbose', '-v', default=logging.INFO,
                           action="store_const", dest="loglevel", const=logging.DEBUG,
                           help="set loglevel to DEBUG")
    log_group.add_argument('--quiet', '-q',
                           action="store_const", dest="loglevel", const=logging.ERROR,
                           help="set loglevel to ERROR")

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(name)s/%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=args.loglevel
    )

    # Setup environment

    ## Disable downloads from hf
    os.environ["HF_HUB_OFFLINE"] = "1"

    ## Unitxt need to set this to run certain benchmarks
    os.environ["UNITXT_ALLOW_UNVERIFIED_CODE"] = "True"

    ## If we don't set this then Unitxt
    ## will redownload the catalog each run
    os.environ["UNITXT_USE_ONLY_LOCAL_CATALOGS"] = "True"

    ##
    os.environ["UNITXT_ARTIFACTORIES"] = args.catalog_path

    args.tasks = args.tasks.split(',')

    logger.info("CLI called with " + str(vars(args)))

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
        exec_lm_eval(**vars(args))


if __name__ == '__main__':
    eval_cli()
