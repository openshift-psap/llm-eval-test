# llm-eval-test

A wrapper around [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [Unitxt](https://github.com/IBM/unitxt) designed for evaluation of a local inference endpoint.

## Requirements

### To Install

- Python 3.9 or newer

### To Run

- An OpenAI API-compatible inference server; like [vLLM](https://github.com/vllm-project/vllm)
- A directory containing the necessary datasets for the benchmark (see example)

### To Develop

- [poetry](https://python-poetry.org/docs/#installation)

## Getting Started

``` sh
# Create a Virtual Environment
python -m venv venv
source venv/bin/activate

# Install the package
pip install git+https://github.com/sjmonson/llm-eval-test.git

# View run options
llm-eval-test run --help
```

## Download Usage

``` sh

# Create dataset directory
DATASETS_DIR=$(pwd)/datasets
mkdir $DATASETS_DIR

#To download the datasets:

llm-eval-test download -h
usage: llm-eval-test download [-h] [--catalog-path PATH] [--tasks-path PATH] [--offline | --no-offline] [-v | -q] -t TASKS [-d DATASETS] [-f | --force-download | --no-force-download]

download datasets for open-llm-v1 tasks

options:
  -h, --help            show this help message and exit

  -t TASKS, --tasks TASKS
                        comma separated tasks to download for example: arc_challenge,hellaswag (default: None)
  -d DATASETS, --datasets DATASETS
                        Dataset directory (default: ./datasets)
  -f, --force-download, --no-force-download
                        Force download datasets even it already exist (default: False)


llm-eval-test download --tasks arc_challenge,GSM8K,HellaSwag
llm-eval-test download --tasks leaderboard
llm-eval-test download --tasks arc_challenge,GSM8K,HellaSwag -f (to overwrite the previously downloaded datasets)
```

## Run Usage

```
usage: llm-eval-test run [-h] [--catalog_path PATH] [--tasks_path PATH] [-v | -q]
-H ENDPOINT -m MODEL -t TASKS -d PATH [-b INT] [-o OUTPUT]

Run tasks

options:
  -h, --help            show this help message and exit
  --catalog_path PATH   unitxt catalog directory
  --tasks_path PATH     lm-eval tasks directory
  -v, --verbose         set loglevel to DEBUG
  -q, --quiet           set loglevel to ERROR
  -b INT, --batch_size INT
                        per-request batch size
  -o OUTPUT, --output OUTPUT
                        results output file

required:
  -H ENDPOINT, --endpoint ENDPOINT
                        OpenAI API-compatible endpoint
  -m MODEL, --model MODEL
                        name of the model under test
  -t TASKS, --tasks TASKS
                        comma separated list of tasks
  -d PATH, --datasets PATH
                        path to dataset storage

```

### Example: MMLU-Pro Benchmark

``` sh
# Create dataset storage
DATASETS_DIR=$(pwd)/datasets
mkdir $DATASETS_DIR

# Download the MMLU-Pro dataset
DATASET=TIGER-Lab/MMLU-Pro

# Use your preferred method of downloading the dataset to $DATASETS_DIR/$DATASET
# to use the huggingface-cli:
pip install huggingface_hub[cli]
huggingface-cli download $DATASET --repo-type dataset --local-dir $DATASETS_DIR/$DATASET

# Run the benchmark
ENDPOINT=http://127.0.0.1:8080/v1/completions # An OpenAI API-compatable completions endpoint
MODEL_NAME=meta-llama/Llama-3.1-8B # Name of the model hosted on the inference server
llm-eval-test run --endpoint $ENDPOINT --model $MODEL_NAME --datasets $DATASETS_DIR --tasks mmlu_pro
```

