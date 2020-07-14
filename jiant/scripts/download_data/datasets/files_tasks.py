import os

import jiant.scripts.download_data.utils as download_utils
import jiant.utils.python.io as py_io


def download_squad_v1_data_and_write_config(task_data_path: str, task_config_path: str):
    os.makedirs(task_data_path, exist_ok=True)
    train_path = os.path.join(task_data_path, "train-v1.1.json")
    val_path = os.path.join(task_data_path, "dev-v1.1.json")
    download_utils.download_file(
        url="https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
        file_path=train_path,
    )
    download_utils.download_file(
        url="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json", file_path=val_path,
    )
    py_io.write_json(
        data={
            "task": "squad",
            "paths": {"train": train_path, "val": val_path},
            "version_2_with_negative": False,
            "name": "squad_v1",
        },
        path=task_config_path,
    )


def download_squad_v2_data_and_write_config(task_data_path: str, task_config_path: str):
    os.makedirs(task_data_path, exist_ok=True)
    train_path = os.path.join(task_data_path, "train-v2.0.json")
    val_path = os.path.join(task_data_path, "dev-v2.0.json")
    download_utils.download_file(
        url="https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
        file_path=train_path,
    )
    download_utils.download_file(
        url="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json", file_path=val_path,
    )
    py_io.write_json(
        data={
            "task": "squad",
            "paths": {"train": train_path, "val": val_path},
            "version_2_with_negative": True,
            "name": "squad_v2",
        },
        path=task_config_path,
    )
