"""Translate raw prediction files for SuperGLUE tasks into format expected by SuperGLUE leaderboard.

This script helps translate the raw prediction files for SuperGLUE tasks into the jsonl files
required by the SuperGLUE leaderboard. See https://super.gluebenchmark.com/faq for leaderboard info.

To use this script specify one or more raw prediction input filepath in raw_pred_input_filepaths.

"""
import json
import torch

from jiant.tasks import retrieval

# for each SuperGLUE task, provide a filepath for the raw prediction file as a value in this map:
raw_pred_input_filepaths = {
    "boolq": None,
    "cb": None,
    "copa": None,
    "mrc": None,
    "record": None,
    "rte": None,
    "wic": None,
    "wsc": None,
    # "TBD1": None, # Broadcoverage diagnostic
    # "TBD2": None, # Winogender Schema diagnostic
}

# this map specifies the location where the SuperGLUE-leaderboard-submission-formatted prediction
# outputs should be saved. By default the files will be saved in the current directory with the
# filenames expected by the leaderboard.
formatted_pred_output_filepaths = {
    "boolq": "./BoolQ.jsonl",
    "cb": "./CB.jsonl",
    "copa": "./COPA.jsonl",
    "mrc": "./MultiRC.jsonl",
    "record": "./ReCoRD.jsonl",
    "rte": "./RTE.jsonl",
    "wic": "./WiC.jsonl",
    "wsc": "./WSC.jsonl",
    # "TBD1": "./AX-b.jsonl",
    # "TBD2": "./AX-g.jsonl",
}

for task_name, input_filepath in raw_pred_input_filepaths.items():
    if input_filepath:
        task = retrieval.get_task_class(task_name)
        task_preds = torch.load(input_filepath)[task_name]
        formatted_preds = task.super_glue_format_preds(task_preds)
        with open(formatted_pred_output_filepaths[task_name], "w") as f:
            for entry in formatted_preds:
                json.dump(entry, f)
                f.write("\n")
