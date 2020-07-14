"""
Downloading adapted from
https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/
    17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
"""
import csv
import os
import shutil
import urllib.request
import zipfile

import jiant.utils.python.io as py_io


GLUE_CONVERSION = {
    "cola": {
        "data": {
            "train": {"cols": {"text": 3, "label": 1}},
            "val": {"cols": {"text": 3, "label": 1}, "meta": {"filename": "dev"}},
            "test": {"cols": {"text": 1}, "meta": {"skiprows": 1}},
        },
        "dir_name": "CoLA",
    },
    "mnli": {
        "data": {
            "train": {
                "cols": {"premise": 8, "hypothesis": 9, "label": 11},
                "meta": {"skiprows": 1},
            },
            "val": {
                "cols": {"premise": 8, "hypothesis": 9, "label": 15},
                "meta": {"filename": "dev_matched", "skiprows": 1},
            },
            "val_mismatched": {
                "cols": {"premise": 8, "hypothesis": 9, "label": 15},
                "meta": {"filename": "dev_mismatched", "skiprows": 1},
            },
            "test": {
                "cols": {"premise": 8, "hypothesis": 9},
                "meta": {"filename": "test_matched", "skiprows": 1},
            },
            "test_mismatched": {
                "cols": {"premise": 8, "hypothesis": 9},
                "meta": {"filename": "test_mismatched", "skiprows": 1},
            },
        },
        "dir_name": "MNLI",
    },
    "mrpc": {
        "data": {
            "train": {"cols": {"text_a": 3, "text_b": 4, "label": 0}, "meta": {"skiprows": 1}},
            "val": {
                "cols": {"text_a": 3, "text_b": 4, "label": 0},
                "meta": {"filename": "dev", "skiprows": 1},
            },
            "test": {"cols": {"text_a": 3, "text_b": 4}, "meta": {"skiprows": 1}},
        },
        "dir_name": "MRPC",
    },
    "qnli": {
        "data": {
            "train": {"cols": {"premise": 1, "hypothesis": 2, "label": 3}, "meta": {"skiprows": 1}},
            "val": {
                "cols": {"premise": 1, "hypothesis": 2, "label": 3},
                "meta": {"filename": "dev", "skiprows": 1},
            },
            "test": {"cols": {"premise": 1, "hypothesis": 2}, "meta": {"skiprows": 1}},
        },
        "dir_name": "QNLI",
    },
    "qqp": {
        "data": {
            "train": {"cols": {"text_a": 3, "text_b": 4, "label": 5}, "meta": {"skiprows": 1}},
            "val": {
                "cols": {"text_a": 3, "text_b": 4, "label": 5},
                "meta": {"filename": "dev", "skiprows": 1},
            },
            "test": {"cols": {"text_a": 1, "text_b": 2}, "meta": {"skiprows": 1}},
        },
        "dir_name": "QQP",
    },
    "rte": {
        "data": {
            "train": {"cols": {"premise": 1, "hypothesis": 2, "label": 3}, "meta": {"skiprows": 1}},
            "val": {
                "cols": {"premise": 1, "hypothesis": 2, "label": 3},
                "meta": {"filename": "dev", "skiprows": 1},
            },
            "test": {"cols": {"premise": 1, "hypothesis": 2}, "meta": {"skiprows": 1}},
        },
        "dir_name": "RTE",
    },
    "snli": {
        "data": {
            "train": {
                "cols": {"premise": 7, "hypothesis": 8, "label": 10},
                "meta": {"skiprows": 1},
            },
            "val": {
                "cols": {"premise": 7, "hypothesis": 8, "label": 14},
                "meta": {"filename": "dev", "skiprows": 1},
            },
            "test": {"cols": {"premise": 7, "hypothesis": 8, "label": 14}, "meta": {"skiprows": 1}},
        },
        "dir_name": "SNLI",
    },
    "sst": {
        "data": {
            "train": {"cols": {"text": 0, "label": 1}, "meta": {"skiprows": 1}},
            "val": {"cols": {"text": 0, "label": 1}, "meta": {"filename": "dev", "skiprows": 1}},
            "test": {"cols": {"text": 1}, "meta": {"skiprows": 1}},
        },
        "dir_name": "SST-2",
    },
    "stsb": {
        "data": {
            "train": {"cols": {"text_a": 7, "text_b": 8, "label": 9}, "meta": {"skiprows": 1}},
            "val": {
                "cols": {"text_a": 7, "text_b": 8, "label": 9},
                "meta": {"filename": "dev", "skiprows": 1},
            },
            "test": {"cols": {"text_a": 7, "text_b": 8}, "meta": {"skiprows": 1}},
        },
        "dir_name": "STS-B",
    },
    "wnli": {
        "data": {
            "train": {"cols": {"premise": 1, "hypothesis": 2, "label": 3}, "meta": {"skiprows": 1}},
            "val": {
                "cols": {"premise": 1, "hypothesis": 2, "label": 3},
                "meta": {"filename": "dev", "skiprows": 1},
            },
            "test": {"cols": {"premise": 1, "hypothesis": 2}, "meta": {"skiprows": 1}},
        },
        "dir_name": "WNLI",
    },
    "glue_diagnostics": {
        "data": {
            "test": {
                "cols": {"premise": 1, "hypothesis": 2},
                "meta": {"filename": "diagnostic", "skiprows": 1},
            },
        },
        "dir_name": "diagnostic",
    },
}

TASK2PATH = {
    "cola": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/"
    "data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4",
    "sst": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/"
    "data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8",
    "mrpc": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/"
    "data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc",
    "qqp": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/"
    "data%2FQQP.zip?alt=media&token=700c6acf-160d-4d89-81d1-de4191d02cb5",
    "stsb": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/"
    "data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5",
    "mnli": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/"
    "data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce",
    "snli": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/"
    "data%2FSNLI.zip?alt=media&token=4afcfbb2-ff0c-4b2d-a09a-dbf07926f4df",
    "qnli": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/"
    "data%2FQNLIv2.zip?alt=media&token=6fdcf570-0fc5-4631-8456-9505272d1601",
    "rte": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/"
    "data%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb",
    "wnli": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/"
    "data%2FWNLI.zip?alt=media&token=068ad0a0-ded7-4bd7-99a5-5e00222e0faf",
    "glue_diagnostics": "https://storage.googleapis.com/mtl-sentence-representations.appspot.com/"
    "tsvsWithoutLabels%2FAX.tsv?GoogleAccessId=firebase-adminsdk-0khhl@"
    "mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature"
    "=DuQ2CSPt2Yfre0C%2BiISrVYrIFaZH1Lc7hBVZDD4ZyR7fZYOMNOUGpi8QxBmTNOrNPjR3z1cggo7WXFf"
    "rgECP6FBJSsURv8Ybrue8Ypt%2FTPxbuJ0Xc2FhDi%2BarnecCBFO77RSbfuz%2Bs95hRrYhTnByqu3U"
    "%2FYZPaj3tZt5QdfpH2IUROY8LiBXoXS46LE%2FgOQc%2FKN%2BA9SoscRDYsnxHfG0IjXGwHN"
    "%2Bf88q6hOmAxeNPx6moDulUF6XMUAaXCSFU%2BnRO2RDL9CapWxj%2BDl7syNyHhB7987hZ80B"
    "%2FwFkQ3MEs8auvt5XW1%2Bd4aCU7ytgM69r8JDCwibfhZxpaa4gd50QXQ%3D%3D",
}

MRPC_TRAIN = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt"
MRPC_TEST = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt"


def download_and_extract(task_name: str, task_data_path: str):
    """Download raw GLUE task data (except MRPC, diagnostic)"""
    data_file = os.path.join(task_data_path, f"{task_name}.zip")
    urllib.request.urlretrieve(TASK2PATH[task_name], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(task_data_path)
    os.remove(data_file)


def download_mrpc(task_data_path: str):
    """Download raw MRPC data"""
    mrpc_dir = os.path.join(task_data_path, "MRPC")
    if not os.path.isdir(mrpc_dir):
        os.mkdir(mrpc_dir)
    mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
    mrpc_test_file = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
    urllib.request.urlretrieve(MRPC_TRAIN, mrpc_train_file)
    urllib.request.urlretrieve(MRPC_TEST, mrpc_test_file)
    urllib.request.urlretrieve(TASK2PATH["mrpc"], os.path.join(mrpc_dir, "dev_ids.tsv"))

    dev_ids = []
    with open(os.path.join(mrpc_dir, "dev_ids.tsv"), encoding="utf8") as ids_fh:
        for row in ids_fh:
            dev_ids.append(row.strip().split("\t"))

    with open(mrpc_train_file, encoding="utf8") as data_fh, open(
        os.path.join(mrpc_dir, "train.tsv"), "w", encoding="utf8"
    ) as train_fh, open(os.path.join(mrpc_dir, "dev.tsv"), "w", encoding="utf8") as dev_fh:
        header = data_fh.readline()
        train_fh.write(header)
        dev_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split("\t")
            if [id1, id2] in dev_ids:
                dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
            else:
                train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))

    with open(mrpc_test_file, encoding="utf8") as data_fh, open(
        os.path.join(mrpc_dir, "test.tsv"), "w", encoding="utf8"
    ) as test_fh:
        _header = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split("\t")
            test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))


def download_diagnostic(data_dir: str):
    """Download raw GLUE diagnostic data"""
    os.makedirs(os.path.join(data_dir, "diagnostic"), exist_ok=True)
    data_file = os.path.join(data_dir, "diagnostic", "diagnostic.tsv")
    urllib.request.urlretrieve(TASK2PATH["glue_diagnostics"], data_file)


def read_tsv(input_file, quotechar=None, skiprows=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        result = list(csv.reader(f, delimiter="\t", quotechar=quotechar))
    if skiprows:
        result = result[skiprows:]
    return result


def download_glue_data(task_name: str, task_data_path: str) -> str:
    """Download raw GLUE data

    Args:
        task_name: Task name
        task_data_path: Path to write task data to

    Returns:
        Path to *raw* task data folder, intended to be deleted later.
    """
    if task_name == "mrpc":
        download_mrpc(task_data_path)
    elif task_name == "glue_diagnostics":
        download_diagnostic(task_data_path)
    else:
        download_and_extract(task_name=task_name, task_data_path=task_data_path)
    return os.path.join(task_data_path, GLUE_CONVERSION[task_name]["dir_name"])


def get_full_examples(task_name: str, raw_task_data_path: str) -> dict:
    """Get examples from raw task data

    Args:
        task_name: GLUE task name
        raw_task_data_path: directory containing raw data

    Returns:
        Dict of list of examples (dicts)
    """
    task_metadata = GLUE_CONVERSION[task_name]
    all_examples = {}
    for phase, phase_config in task_metadata["data"].items():
        meta_dict = phase_config.get("meta", {})
        filename = meta_dict.get("filename", phase)
        rows = read_tsv(
            os.path.join(raw_task_data_path, f"{filename}.tsv"), skiprows=meta_dict.get("skiprows"),
        )
        examples = []
        for row in rows:
            try:
                example = {}
                for col, i in phase_config["cols"].items():
                    example[col] = row[i]
                examples.append(example)
            except IndexError:
                if task_name == "qqp":
                    continue
        all_examples[phase] = examples
    return all_examples


def convert_glue_data_to_jsonl(
    raw_task_data_path: str, task_data_path: str, task_name: str
) -> dict:
    """Convert raw GLUE data to jsonl for one task

    Args:
        raw_task_data_path: Path to raw GLUE data directory
        task_data_path: Path to write .jsonl GLUE data
        task_name: task name

    Returns:
        dictionary to paths of .jsonl GLUE data
    """
    os.makedirs(task_data_path, exist_ok=True)
    task_all_examples = get_full_examples(
        task_name=task_name, raw_task_data_path=raw_task_data_path,
    )
    paths_dict = {}
    for phase, phase_data in task_all_examples.items():
        phase_data_path = os.path.join(task_data_path, f"{phase}.jsonl")
        py_io.write_jsonl(
            data=phase_data, path=phase_data_path,
        )
        paths_dict[phase] = phase_data_path
    return paths_dict


def download_glue_data_and_write_config(
    task_name: str, task_data_base_path: str, task_config_base_path: str
):
    """Download GLUE data, convert to jsonl, delete raw data, and write config (for one task)

    For task_name="mnli", this will write both mnli_config.json and mnli_mismatched_config.json

    Args:
        task_name: Task name
        task_data_base_path: base path to write task data into
        task_config_base_path: base path to write configs into
    """
    task_data_path = os.path.join(task_data_base_path, task_name)
    os.makedirs(task_data_path, exist_ok=True)
    os.makedirs(task_config_base_path, exist_ok=True)
    raw_task_data_path = download_glue_data(task_name=task_name, task_data_path=task_data_path)
    paths_dict = convert_glue_data_to_jsonl(
        raw_task_data_path=raw_task_data_path, task_data_path=task_data_path, task_name=task_name,
    )
    if task_name == "mnli":
        py_io.write_json(
            data={
                "task": "mnli",
                "paths": {
                    "train": paths_dict["train"],
                    "val": paths_dict["val"],
                    "test": paths_dict["test"],
                },
                "name": "mnli",
            },
            path=os.path.join(task_config_base_path, f"mnli_config.json"),
        )
        py_io.write_json(
            data={
                "task": "mnli",
                "paths": {
                    "val": paths_dict["val_mismatched"],
                    "test": paths_dict["test_mismatched"],
                },
                "name": "mnli_mismatched",
            },
            path=os.path.join(task_config_base_path, f"mnli_mismatched_config.json"),
        )
    elif task_name == "glue_diagnostics":
        py_io.write_json(
            data={"task": "mnli", "paths": paths_dict, "name": task_name},
            path=os.path.join(task_config_base_path, f"{task_name}_config.json"),
        )
    else:
        py_io.write_json(
            data={"task": task_name, "paths": paths_dict, "name": task_name},
            path=os.path.join(task_config_base_path, f"{task_name}_config.json"),
        )
    shutil.rmtree(raw_task_data_path)
