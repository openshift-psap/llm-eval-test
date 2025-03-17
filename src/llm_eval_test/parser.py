import enum
import os
import logging
import argparse
import datetime


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
    parser_base.add_argument('--offline', type=bool,
                             default=True, action=argparse.BooleanOptionalAction,
                             help="Disable/enable updating datasets from the internet")
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
    parser_run.add_argument('--tokenizer',
                          help="path or huggingface tokenizer name, if none uses model name")
    parser_run.add_argument('-b', '--batch', default=Defaults.batch_size, type=int,
                            help="per-request batch size", metavar='INT')
    parser_run.add_argument('-r', '--retry', default=Defaults.retry_count, type=int,
                            help="max number of times to retry a single request", metavar='INT')
    now_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    output_group = parser_run.add_mutually_exclusive_group()
    output_group.add_argument('-o', '--output', type=argparse.FileType('w'),
                              default=f"{work_dir}/{now_time}.json",
                              help="results output file")
    output_group.add_argument('--no-output',
                              action="store_const", dest="output", const=None,
                              help="disable results output file")
    parser_run.add_argument('--format', type=OutputFormat, default=OutputFormat.default,
                              choices=list(OutputFormat),
                              help="format of output file")

    parser_list = subparsers.add_parser(
        'list',
        description="List available tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parser_base]
    )
    parser_download = subparsers.add_parser(
        'download',
        description="download datasets for open-llm-v1 tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parser_base]
    )
    parser_download.add_argument('-t', '--tasks',
                                 type=str, required=True, help="comma separated tasks to download for example: arc_challenge,hellaswag")
    parser_download.add_argument("-d", "--datasets", type=dir_path, default=f"{work_dir}/datasets", help="Dataset directory")
    parser_download.add_argument("-f", "--force-download", action=argparse.BooleanOptionalAction, type=bool, default=False, help="Force download datasets even it already exist")
    return parser
