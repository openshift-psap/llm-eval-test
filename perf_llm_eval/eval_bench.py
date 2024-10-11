#!/usr/bin/env python3

import os
import argparse
import logging
import dotenv
from lm_eval.evaluator import simple_evaluate
from lm_eval.tasks import TaskManager  # type: ignore

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

def exec_lm_eval(tasks, model, endpoint, **kwargs):

    model_args = dict(
        model = model,
        base_url = endpoint,
        num_concurent=1,
        max_retries=3,
        tokenizer_backend=None,
        tokenized_requests=False
    )

    model_args_str = ','.join([f"{k}={repr(v)}" for k,v in model_args.items()])
    tm = TaskManager(verbosity="DEBUG", include_path=kwargs['tasks_path'])

    results = simple_evaluate(
        model="local-completions",
        model_args=model_args_str,
        tasks=tasks,
        #num_fewshot=self.few_shots,
        batch_size=kwargs['batch_size'],
        task_manager=tm,
    )

    print(results)


def eval_cli():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
                    description='Helper script for running PSAP relevent LLM benchmarks.',
                    epilog='')

    local_dir = f"{os.path.dirname(__file__)}/lm_eval"
    parser.add_argument('--catalog_path',
                        default=f"{local_dir}/catalog",
                        help="unitxt catalog directory")
    parser.add_argument('--tasks_path',
                        default=f"{local_dir}/tasks",
                        help="unitxt catalog directory")
    cache_dir = os.environ.get("XDG_CACHE_DIR", f"{os.environ['HOME']}/.cache")
    parser.add_argument('--cache_path',
                        default=f"{cache_dir}/huggingface_eval",
                        help="cache directory")
    parser.add_argument('--endpoint', '-H', required=True,
                        default='http://127.0.0.1:8000/v1/completions',
                        help='OpenAI API-compatible endpoint')
    parser.add_argument('--model', '-m', required=True,
                        help="name of the model under test")
    parser.add_argument('--batch_size', '-b', default=64, type=int,
                        help="")
    parser.add_argument('--tasks', '-t', required=True,
                        help="Comma separated list of tasks")

    args = parser.parse_args()

    # Setup environment

    ## Disable downloads from hf
    os.environ["HF_HUB_OFFLINE"] = "1"

    ## Unitxt need to set this to run certain benchmarks
    os.environ["UNITXT_ALLOW_UNVERIFIED_CODE"] = "True"

    ## If we don't set this then Unitxt
    ## will redownload the catalog each run
    os.environ["UNITXT_USE_ONLY_LOCAL_CATALOGS"] = "True"

    ## Set our own HF_HOME for dataset (and unitxt) storage
    os.environ["HF_HOME"] = args.cache_path

    ##
    os.environ["UNITXT_ARTIFACTORIES"] = args.catalog_path

    args.tasks = args.tasks.split(',')

    logger.info("Called with " + str(vars(args)))

    exec_lm_eval(**vars(args))

if __name__ == '__main__':
    eval_cli()
