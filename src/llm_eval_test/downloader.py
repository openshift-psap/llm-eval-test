import logging
import os
import tempfile

from lm_eval.api.group import ConfigurableGroup
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager

logger = logging.getLogger("downloader")


def download_datasets(datasets_dir: str, tasks: list[str], tasks_path: str, force_download: bool = False) -> dict:
    task_list = [tasks] if isinstance(tasks, str) else tasks

    # TaskManager
    tm = TaskManager(
        include_path=tasks_path,
        include_defaults=True,
        verbosity=logging.getLevelName(logger.level),
    )
    # Load tasks and groups
    task_dict = tm.load_task_or_group(task_list)
    if not task_dict:
        logger.error(f"No tasks loaded for {task_list}")
        return {}
    # Map tasks and subtasks to datasets
    task_to_dataset = {}
    for group_or_task_obj, subtasks_or_task in task_dict.items():
        # Handle the top-level object (could be group or task)
        if isinstance(group_or_task_obj, ConfigurableGroup | ConfigurableTask):
            task_name = group_or_task_obj._config.get("group", None) or group_or_task_obj._config.get("task", "unknown")
            if isinstance(group_or_task_obj, ConfigurableGroup):
                sub_result = process_task_object(subtasks_or_task, task_name)
                task_to_dataset.update(sub_result)
        elif isinstance(group_or_task_obj, str):
            # if task is standalone type str i.e arc_challenge, arc_easy, etc
            sub_result = process_task_object(subtasks_or_task, group_or_task_obj)
            task_to_dataset.update(sub_result)
        else:
            logger.warning(f"!! Unexpected key type in task_dict: {type(group_or_task_obj)}")

    logger.info(f"Task mapping to datasets name =>: {task_to_dataset}")
    os.makedirs(datasets_dir, exist_ok=True)

    # Download datasets
    local_paths = {}
    for task_name, dataset_repo in task_to_dataset.items():
        target_dir = os.path.join(datasets_dir, dataset_repo)  # eg: 'allenai/ai2_arc/ARC-Challenge'
        logger.info(f"Downloading '{task_name}' dataset from {dataset_repo} to {target_dir}")
        try:
            if force_download or not os.path.exists(target_dir):
                from huggingface_hub import snapshot_download

                with tempfile.TemporaryDirectory() as tmpdir:
                    snapshot_download(
                        repo_id=dataset_repo,
                        repo_type="dataset",
                        cache_dir=tmpdir,
                        local_dir=target_dir,
                        local_dir_use_symlinks=False,  # TODO: Remove as depercated
                        force_download=True,
                        token=os.getenv("HF_TOKEN", True),  # Str or True
                    )
                    local_paths[task_name] = target_dir
        except Exception as e:
            logger.error(f"Failed to download '{task_name}' from {dataset_repo}: {e}")

    return local_paths


def process_task_object(task_obj, task_name=None) -> dict:
    """ """
    task_to_dataset = {}
    if isinstance(task_obj, ConfigurableTask) or isinstance(task_obj, str):
        # Individual task
        dataset_path = getattr(task_obj, "DATASET_PATH", None)
        if dataset_path:
            task_to_dataset[task_name] = dataset_path
            logger.info(f"Task '{task_name}' mapped to {dataset_path}")
        else:
            logger.warning(f"No DATASET_PATH for task '{task_name}'")

    elif isinstance(task_obj, dict):
        # Nested dictionary of tasks/groups
        for sub_name, sub_obj in task_obj.items():
            sub_result = process_task_object(sub_obj, sub_name)
            task_to_dataset.update(sub_result)

    return task_to_dataset
