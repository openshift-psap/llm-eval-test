# perf-llm-eval

A wrapper around [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [Unitxt](https://github.com/IBM/unitxt) designed for evaluation of a local inference endpoint.

## Requirements

### Install

- Python 3.9 or newer

### Run

- An OpenAI API-compatible inference server; like [vLLM](https://github.com/vllm-project/vllm)
- A directory containing the necessary datasets for the benchmark (see example)

### Develop

- [poetry](https://python-poetry.org/docs/#installation)

## Getting Started

``` sh
# Create a Virtual Environment
python -m venv venv
source venv/bin/activate

# Install the package
pip install https://github.com/sjmonson/perf-llm-eval.git

# Run
perf-llm-eval --help
```

## Usage

```
usage: perf-llm-eval [-h] [--catalog_path CATALOG_PATH] [--tasks_path TASKS_PATH] --datasets DATASETS --endpoint ENDPOINT --model MODEL [--batch_size BATCH_SIZE] --tasks TASKS

Helper script for running PSAP relevent LLM benchmarks.

optional arguments:
  -h, --help            show this help message and exit
  --catalog_path CATALOG_PATH
                        unitxt catalog directory
  --tasks_path TASKS_PATH
                        unitxt catalog directory
  --datasets DATASETS, -d DATASETS
                        path to dataset storage
  --endpoint ENDPOINT, -H ENDPOINT
                        OpenAI API-compatible endpoint
  --model MODEL, -m MODEL
                        name of the model under test
  --batch_size BATCH_SIZE, -b BATCH_SIZE
  --tasks TASKS, -t TASKS
                        Comma separated list of tasks
```

### Example: MMLU-Pro Benchmark

``` sh
# Create dataset storage
export DATASETS_DIR=$(pwd)/datasets
mkdir $DATASETS_DIR

# Download the MMLU-Pro dataset
export DATASET=TIGER-Lab/MMLU-Pro
huggingface-cli download $DATASET --repo-type dataset --local-dir $DATASETS_DIR/$DATASET

# Run the benchmark
export ENDPOINT=http://127.0.0.1:8003/v1/completions # An OpenAI API-compatable completions endpoint
export MODEL_NAME=meta-llama/Llama-3.1-8B # Name of the model hosted on the inference server
perf-llm-eval --endpoint $ENDPOINT --model $MODEL_NAME --datasets $DATASET_DIR --tasks mmlu_pro_all
```
