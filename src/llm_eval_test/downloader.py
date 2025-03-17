from huggingface_hub import snapshot_download
from lm_eval.tasks import TaskManager  # type: ignore
from lm_eval.api.task import Task # type: ignore
import os 
import logging
logger = logging.getLogger("downloader")


def download_datasets(datasets_dir: str, tasks: list[str], tasks_path: str, force_download: bool = False):

    task_list = [tasks] if isinstance(tasks, str) else tasks
    tm = TaskManager(
        include_path=tasks_path,
        include_defaults=True,
        verbosity=logging.getLevelName(logger.level),
    )

    task_dict = tm.load_task_or_group(task_list)
    if not task_dict:
        logger.error(f"No tasks loaded for {task_list}")
        return {}
    task_to_dataset = {}
    for task_name, task_obj in task_dict.items():
        dataset_path = getattr(task_obj, "DATASET_PATH", None)
        dataset_name = getattr(task_obj, "DATASET_NAME", None)
        if dataset_path:
            # Construct full dataset identifier (e.g., "allenai/ai2_arc" or "hellaswag")
            task_to_dataset[task_name] = dataset_path
        else:
            logger.warning(f"No DATASET_PATH found for task '{task_name}'")

    logger.info(f"Task mapping to datasets name =>: {task_to_dataset} {dataset_name}")
    os.makedirs(datasets_dir, exist_ok=True)

    # Download datasets
    local_paths = {}
    for task_name, dataset_repo in task_to_dataset.items():
        target_dir = os.path.join(datasets_dir, dataset_repo) # eg: 'allenai/ai2_arc/ARC-Challenge'
        logger.info(f"Downloading '{task_name}' dataset from {dataset_repo} to {target_dir}")
        try:
            if force_download or not os.path.exists(target_dir):
                snapshot_download(
                    repo_id=dataset_repo,
                    repo_type="dataset",
                    local_dir=target_dir,
                    local_dir_use_symlinks=False,
                    use_auth_token=os.getenv("HF_TOKEN")
                )
                local_paths[task_name] = target_dir
        except Exception as e:
            logger.error(f"Failed to download '{task_name}' from {dataset_repo}: {e}")
    
    return local_paths


