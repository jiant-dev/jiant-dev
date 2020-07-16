import os

import jiant.scripts.download_data.utils as download_utils
import jiant.utils.python.io as py_io

SQUAD_PREFIX = "squad_v"


def download_squad_data_and_write_config(
    task_name: str, task_data_path: str, task_config_path: str
):
    assert task_name.startswith(SQUAD_PREFIX)
    version = float(task_name[len(SQUAD_PREFIX) :])

    os.makedirs(task_data_path, exist_ok=True)
    train_path = os.path.join(task_data_path, "train-v" + str(version) + ".json")
    val_path = os.path.join(task_data_path, "dev-v" + str(version) + ".json")
    download_utils.download_file(
        url="https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v" + str(version) + ".json",
        file_path=train_path,
    )
    download_utils.download_file(
        url="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v" + str(version) + ".json",
        file_path=val_path,
    )
    py_io.write_json(
        data={
            "task": "squad",
            "paths": {"train": train_path, "val": val_path},
            "version_2_with_negative": int(version) != 2,
            "name": "squad_v2",
        },
        path=task_config_path,
    )
