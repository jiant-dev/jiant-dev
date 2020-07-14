import os
import urllib
import zipfile

import jiant.utils.python.io as py_io


_STANDARD_FILENAMES = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}

SUPERGLUE_METADATA = {
    "superglue_broadcoverage_diagnostics": {
        "url": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/AX-b.zip",
        "filenames": {
            "test": "AX-B.jsonl",
        },
        "jiant_task": "rte",
    },
    "cb": {
        "url": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/CB.zip",
        "filenames": _STANDARD_FILENAMES,
    },
    "copa": {
        "url": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/COPA.zip",
        "filenames": _STANDARD_FILENAMES,
    },
    "multirc": {
        "url": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/MultiRC.zip",
        "filenames": _STANDARD_FILENAMES,
    },
    "rte": {
        "url": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip",
        "filenames": _STANDARD_FILENAMES,
    },
    "wic": {
        "url": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WiC.zip",
        "filenames": _STANDARD_FILENAMES,
    },
    "wsc": {
        "url": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip",
        "filenames": _STANDARD_FILENAMES,
    },
    "boolq": {
        "url": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip",
        "filenames": _STANDARD_FILENAMES,
    },
    "record": {
        "url": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip",
        "filenames": _STANDARD_FILENAMES,
    },
    "superglue_winogender_diagnostics": {
        "url": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/AX-g.zip",
        "filenames": {
            "test": "AX-g.jsonl",
        },
        "jiant_task": "wsc",
    },
}


def download_superglue_data(task_name: str, task_data_path: str):
    """Downloads SuperGLUE task data, and returns paths dict

    Args:
        task_name: Task name
        task_data_path: Path to task data

    Returns:
        dictionary to paths of .jsonl SuperGLUE data
    """
    data_file = os.path.join(task_data_path, f"{task_name}.zip")
    urllib.request.urlretrieve(SUPERGLUE_METADATA[task_name]["url"], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(task_data_path)
    os.remove(data_file)
    folder_name = SUPERGLUE_METADATA[task_name]["url"].split("/")[-1].replace(".zip", "")
    paths_dict = {}
    for phase, filename in SUPERGLUE_METADATA[task_name]["filenames"].items():
        path = os.path.join(task_data_path, filename)
        os.rename(
            src=os.path.join(task_data_path, folder_name, filename),
            dst=path,
        )
        paths_dict[phase] = path
    os.rmdir(os.path.join(task_data_path, folder_name))
    return paths_dict


def download_glue_data_and_write_config(
    task_name: str, task_data_path: str, task_config_path: str
):
    """Download SuperGLUE data and write config (for one task)

    Args:
        task_name: Task name
        task_data_path: Path to write task data into
        task_config_path: Path to write configs into
    """
    paths_dict = download_superglue_data(
        task_name=task_name,
        task_data_path=task_data_path,
    )
    task_config = {
        "task": SUPERGLUE_METADATA[task_name].get("jiant_task", task_name),
        "paths": paths_dict,
        "name": task_name,
    }
    py_io.write_json(
        data=task_config,
        path=task_config_path,
    )
