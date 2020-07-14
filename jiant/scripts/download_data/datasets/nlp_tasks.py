"""Use this from tasks that can be obtained from NLP without further/special processing"""

import jiant.scripts.download_data.utils as download_utils
import jiant.utils.python.io as py_io
from jiant.utils.python.strings import replace_prefix


NLP_CONVERSION_DICT = {
    # === GLUE === #
    "cola": {
        "path": "glue",
        "name": "cola",
        "field_map": {"sentence": "text"},
        "label_map": {0: "0", 1: "1"},
    },
    "mnli": {
        "path": "glue",
        "name": "mnli",
        "label_map": {0: "contradiction", 1: "entailment", 2: "neutral"},
        "phase_list": ["train", "validation_matched", "test_matched"],
    },
    "mnli_mismatched": {
        "path": "glue",
        "name": "mnli",
        "label_map": {0: "contradiction", 1: "entailment", 2: "neutral"},
        "phase_list": ["train", "validation_mismatched ", "test_mismatched"],
        "jiant_task_name": "mnli",
    },
    "mrpc": {
        "path": "glue",
        "name": "mrpc",
        "field_map": {"sentence1": "text_a", "sentence2": "text_b"},
        "label_map": {0: "0", 1: "1"},
    },
    "qnli": {
        "path": "glue",
        "name": "qnli",
        "field_map": {"question": "premise", "sentence": "hypothesis"},
        "label_map": {0: "contradiction", 1: "entailment", 2: "neutral"},
    },
    "qqp": {
        "path": "glue",
        "name": "qqp",
        "field_map": {"question1": "text_a", "question2": "text_b"},
        "label_map": {0: "0", 1: "1"},
    },
    "rte": {
        "path": "glue",
        "name": "rte",
        "field_map": {"sentence1": "premise", "sentence2": "hypothesis"},
        "label_map": {0: "entailment", 1: "not_entailment"},
    },
    "sst": {
        "path": "glue",
        "name": "sst2",
        "field_map": {"sentence": "text"},
        "label_map": {0: "0", 1: "1"},
    },
    "stsb": {
        "path": "glue",
        "name": "stsb",
        "field_map": {"sentence1": "text_a", "sentence2": "text_b"},
    },
    "wnli": {
        "path": "glue",
        "name": "wnli",
        "field_map": {"sentence1": "premise", "sentence2": "hypothesis"},
        "label_map": {0: "0", 1: "1"},
    },
    "glue_diagnostics": {
        "path": "glue",
        "name": "ax",
        "jiant_task_name": "mnli",
    },
    # === SuperGLUE === #
    "boolq": {
        "path": "super_glue",
        "name": "boolq",
        "label_map": {0: False, 1: True},
    },
    "cb": {
        "path": "super_glue",
        "name": "cb",
        "label_map": {0: "contradiction", 1: "entailment", 2: "neutral"},
    },
    "copa": {
        "path": "super_glue",
        "name": "copa",
    },
    "multirc": {
        "path": "super_glue",
        "name": "multirc",
    },
    "record": {
        "path": "super_glue",
        "name": "record",
    },
    "wic": {
        "path": "super_glue",
        "name": "wic",
        "label_map": {0: False, 1: True},
    },
    "wsc": {
        "path": "super_glue",
        "name": "wsc.fixed",
        "label_map": {0: False, 1: True},
    },
    "superglue_broadcoverage_diagnostics": {
        "path": "super_glue",
        "name": "axb",
        "field_map": {"sentence1": "premise", "sentence2": "hypothesis"},
        "label_map": {0: "entailment", 1: "not_entailment"},
        "jiant_task_name": "rte",
    },
    "superglue_winogender_diagnostics": {
        "path": "super_glue",
        "name": "axg",
        "label_map": {0: "entailment", 1: "not_entailment"},
        "jiant_task_name": "rte",
    },
    # === Other === #
    "snli": {
        "path": "snli",
        "label_map": {0: "contradiction", 1: "entailment", 2: "neutral"},
    },
}


def download_data_and_write_config(
    task_name: str, task_data_path: str, task_config_path: str
):
    nlp_conversion_metadata = NLP_CONVERSION_DICT[task_name]
    examples_dict = download_utils.convert_nlp_dataset_to_examples(
        path=nlp_conversion_metadata["path"],
        name=nlp_conversion_metadata["name"],
        field_map=nlp_conversion_metadata.get("field_map"),
        label_map=nlp_conversion_metadata.get("label_map"),
        phase_list=nlp_conversion_metadata.get("phase_list"),
    )
    for phase in list(examples_dict):
        if phase.startswith("validation_"):
            examples_dict[replace_prefix(phase, "validation_", "val_")] = examples_dict[phase]
            del examples_dict[phase]
    paths_dict = download_utils.write_examples_to_jsonls(
        examples_dict=examples_dict,
        task_data_path=task_data_path,
    )
    jiant_task_name = nlp_conversion_metadata.get("jiant_task_name", task_name)
    py_io.write_json(
        data={"task": jiant_task_name, "paths": paths_dict, "name": task_name},
        path=task_config_path,
    )
