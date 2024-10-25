

import logging
import json

# Avoid importing these until we want to use lm_eval
from lm_eval.evaluator import simple_evaluate
from lm_eval.tasks import TaskManager  # type: ignore
from lm_eval.utils import handle_non_serializable, make_table

from llm_eval_test.parser import OutputFormat

logger = logging.getLogger("llm-eval-test")

class LMEvalWrapper(object):
    @staticmethod
    def exec(tasks, model, endpoint, **kwargs):
        model_args = dict(
            model = model,
            base_url = endpoint,
            num_concurent=1,
            max_retries=kwargs["retry"],
            tokenizer_backend=None,
            tokenized_requests=False
        )

        model_args_str = ','.join([f"{k}={str(v)}" for k,v in model_args.items()])
        tm = TaskManager(
            include_path=kwargs['tasks_path'],
            include_defaults=False,
            verbosity=logging.getLevelName(logger.level)
        )

        logger.info("Running lm-eval")
        results = simple_evaluate(
            model="local-completions",
            model_args=model_args_str,
            tasks=tasks,
            #num_fewshot=self.few_shots,
            batch_size=kwargs['batch'],
            task_manager=tm,
            verbosity=logging.getLevelName(kwargs['loglevel'])
        )

        if results:
            if kwargs.get('output'):
                # Write results to outfile
                logger.info(f"Writing results to {kwargs['output'].name}")

                if kwargs['format'] == OutputFormat.summary:
                    results_out = results.copy()
                    results_out.pop("samples")
                else: # kwargs['format'] == 'full'
                    results_out = results

                output = json.dumps(
                    results_out, indent=2, default=handle_non_serializable, ensure_ascii=False
                )
                kwargs['output'].write(output)

            # Print output table
            print(make_table(results))
            if results.get('groups'):
                print(make_table(results, "groups"))

    @staticmethod
    def list_tasks(tasks_path: str):
        tm = TaskManager(
            include_path=tasks_path,
            include_defaults=False,
            verbosity=logging.getLevelName(logger.level)
        )

        print(tm.list_all_tasks())
