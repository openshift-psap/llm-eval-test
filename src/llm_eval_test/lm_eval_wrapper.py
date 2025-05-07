import json
import logging
import tempfile

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

# Avoid importing these until we want to use lm_eval
from lm_eval.evaluator import simple_evaluate
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable, make_table
from transformers import AutoTokenizer

from llm_eval_test.parser import OutputFormat

logger = logging.getLogger("llm-eval-test")


class LMEvalWrapper:
    @staticmethod
    def exec(tasks, model, tokenizer, endpoint, **kwargs):
        # Fallback to model if tokenizer is not provided
        tokenizer_repo = tokenizer if tokenizer else model
        chat_template = kwargs.get("chat_template", False)

        # Load the tokenizer to check that it works and the chat template is set if needed
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, use_fast=True)

        # Check if we are missing a chat template when we need one
        if chat_template and not tokenizer.chat_template:
            # HACK: Mistral has a separate file for chat template try to load that on failure
            logger.warning(
                "tokenizer_config.json is missing a chat template. Attempting to load from chat_template.json"
            )
            try:
                file_path = hf_hub_download(repo_id=tokenizer_repo, filename="chat_template.json")
                with open(file_path) as f:
                    template = json.load(f)["chat_template"]
                tokenizer.chat_template = template
            except (json.JSONDecodeError, KeyError) as e:
                raise RuntimeError("Failed to load chat template from alternate location") from e
            except EntryNotFoundError as e:
                raise RuntimeError("No chat template found for given tokenizer") from e

        with tempfile.TemporaryDirectory() as tokenizer_path:
            # Save the modified tokenizer to our temp path
            logger.info(f"Saving tokenizer to {tokenizer_path}")
            tokenizer.save_pretrained(tokenizer_path)

            model_args = {
                "model": model,
                "tokenizer": tokenizer_path,
                "base_url": endpoint,
                "num_concurrent": 1,
                "max_retries": kwargs["retry"],
                "tokenizer_backend": "huggingface",
                "verify_certificate": False,
            }

            model_args_str = ",".join([f"{k}={v!s}" for k, v in model_args.items()])
            tm = TaskManager(
                include_path=kwargs["tasks_path"], include_defaults=False, verbosity=logging.getLevelName(logger.level)
            )

            logger.info("Running lm-eval")
            results = simple_evaluate(
                model="local-completions",
                model_args=model_args_str,
                apply_chat_template=chat_template,
                fewshot_as_multiturn=chat_template,
                tasks=tasks,
                batch_size=kwargs["batch"],
                task_manager=tm,
            )

        if results:
            if kwargs.get("output"):
                # Write results to outfile
                logger.info(f"Writing results to {kwargs['output'].name}")

                if kwargs["format"] == OutputFormat.summary:
                    results_out = results.copy()
                    results_out.pop("samples")
                else:  # kwargs['format'] == 'full'
                    results_out = results

                results_out["let_config"] = kwargs
                output = json.dumps(results_out, indent=2, default=handle_non_serializable, ensure_ascii=False)
                kwargs["output"].write(output)

            # Print output table
            print(make_table(results))
            if results.get("groups"):
                print(make_table(results, "groups"))

    @staticmethod
    def list_tasks(tasks_path: str):
        tm = TaskManager(include_path=tasks_path, include_defaults=False, verbosity=logging.getLevelName(logger.level))

        print(tm.list_all_tasks())
