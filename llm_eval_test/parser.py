

import enum
import os
import logging
import argparse


class OutputFormat(enum.Enum):
    """Log file output format"""
    full    = 'full'
    summary = 'summary'
    default = summary

    def __str__(self):
        return str(self.value)


class Defaults(object):
    """Default values for arguments."""
    batch_size:  int = 32
    retry_count: int = 5
    log_level:   int = logging.INFO


def setup_parser(local_dir: str, work_dir: str) -> argparse.ArgumentParser:
    def dir_path(path: str) -> str:
        ''' Typecheck for directory '''
        if os.path.isdir(path):
            return os.path.abspath(path)
        else:
            raise NotADirectoryError(path)

    parser_base = argparse.ArgumentParser(add_help=False)
    parser_base.add_argument('--catalog-path', type=dir_path,
                        default=f"{local_dir}/benchmarks/catalog",
                             help="unitxt catalog directory", metavar='PATH')
    parser_base.add_argument('--tasks-path', type=dir_path,
                        default=f"{local_dir}/benchmarks/tasks",
                             help="lm-eval tasks directory", metavar='PATH')
    parser_base.add_argument('--format', type=OutputFormat,
                             choices=list(OutputFormat),
                             default=OutputFormat.default,
                             help="format of output file")
    log_group = parser_base.add_mutually_exclusive_group()
    log_group.add_argument('-v', '--verbose', default=Defaults.log_level,
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
    parser_run.add_argument('-b', '--batch', default=Defaults.batch_size, type=int,
                            help="per-request batch size", metavar='INT')
    parser_run.add_argument('-r', '--retry', default=Defaults.retry_count, type=int,
                            help="max number of times to retry a single request", metavar='INT')
    parser_run.add_argument('-o', '--output', type=argparse.FileType('w'),
                            help="results output file")

    parser_list = subparsers.add_parser(
        'list',
        description="List available tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parser_base]
    )

    return parser
