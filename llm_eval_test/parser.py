

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
                        default=f"{local_dir}/benchmarks/catalog",
                             help="unitxt catalog directory", metavar='PATH')
    parser_base.add_argument('--tasks_path', type=dir_path,
                        default=f"{local_dir}/benchmarks/tasks",
                             help="lm-eval tasks directory", metavar='PATH')
    log_group = parser_base.add_mutually_exclusive_group()
    log_group.add_argument('-v', '--verbose', default=logging.INFO,
                           action="store_const", dest="loglevel", const=logging.DEBUG,
                           help="set loglevel to DEBUG")
    log_group.add_argument('-q', '--quiet',
                           action="store_const", dest="loglevel", const=logging.ERROR,
                           help="set loglevel to ERROR")


    parser = argparse.ArgumentParser(
        description='Helper script for running PSAP relevent LLM benchmarks.',
        epilog='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parser_base]
    )
    subparsers = parser.add_subparsers(help='commands', dest='command', required=True)
    parser_run = subparsers.add_parser(
        'run',
        description="Run tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parser_base]
    )
    required = parser_run.add_argument_group('required')
    required.add_argument('-H', '--endpoint', required=True,
                          default='http://127.0.0.1:8000/v1/completions',
                          help='OpenAI API-compatible endpoint')
    required.add_argument('-m', '--model', required=True,
                          help="name of the model under test")
    required.add_argument('-t', '--tasks', required=True,
                          help="comma separated list of tasks")
    required.add_argument('-d', '--datasets', type=dir_path, required=True,
                          default=f"{work_dir}/datasets",
                          help="path to dataset storage", metavar='PATH')
    parser_run.add_argument('-b', '--batch_size', default=64, type=int,
                            help="per-request batch size", metavar='INT')
    parser_run.add_argument('-r', '--retry', default=3, type=int,
                            help="Max number of times to retry a single request", metavar='INT')
    parser_run.add_argument('-o', '--output', type=argparse.FileType('w'),
                            help="results output file")

    parser_list = subparsers.add_parser(
        'list',
        description="List available tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parser_base]
    )

    return parser
