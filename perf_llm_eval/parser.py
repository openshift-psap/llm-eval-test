

import os
import logging
import argparse


def setup_parser(local_dir: str, work_dir: str) -> argparse.ArgumentParser:
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
