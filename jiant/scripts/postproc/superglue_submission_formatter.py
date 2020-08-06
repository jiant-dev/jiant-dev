"""Translate raw prediction files for SuperGLUE tasks into format expected by SuperGLUE leaderboard.

This script translates raw prediction files for SuperGLUE tasks into the jsonl files required
by the SuperGLUE leaderboard. See https://super.gluebenchmark.com/faq for leaderboard info.
"""
import json
import os
import torch

from jiant.tasks import retrieval
from jiant.tasks.constants import SUPERGLUE_TASKS


# this map specifies the location where the SuperGLUE-leaderboard-submission-formatted prediction
# outputs should be saved. By default the files will be saved in the current directory with the
# filenames expected by the leaderboard.
formatted_pred_output_filepaths = {
    "boolq": "BoolQ.jsonl",
    "cb": "CB.jsonl",
    "copa": "COPA.jsonl",
    "multirc": "MultiRC.jsonl",
    "record": "ReCoRD.jsonl",
    "rte": "RTE.jsonl",
    "wic": "WiC.jsonl",
    "wsc": "WSC.jsonl",
    "superglue_broadcoverage_diagnostics": "AX-b.jsonl",
    "superglue_winogender_diagnostics": "AX-g.jsonl",
}

for task_name, input_filepath in raw_pred_input_filepaths.items():
    if input_filepath:
        task = retrieval.get_task_class(task_name)
        task_preds = torch.load(input_filepath)[task_name]
        formatted_preds = task.super_glue_format_preds(task_preds)
        output_filepath = formatted_pred_output_filepaths[task_name]
        with open(output_filepath, "w") as f:
            for entry in formatted_preds:
                json.dump(entry, f)
                f.write("\n")
        print(task_name, ":", os.path.abspath(output_filepath))


def main():
    parser = argparse.ArgumentParser(
        description="Generate formatted output files for SuperGLUE benchmark submission"
    )
    parser.add_argument(
        "--input_base_path",
        required=True,
        help="base input path of SuperGLUE tasks results output by jiant",
    )
    parser.add_argument("--output_path", required=True, help="output path for formatted files")

    args = parser.parse_args()

    # for task_name, input_filepath in raw_pred_input_filepaths.items():
    for task_name in SUPERGLUE_TASKS:
        input_filepath = os.path.join(args.input_base_path, task_name, "test_preds.p")
        output_filepath = os.path.join(args.output_path, formatted_pred_output_filenames[task_name])

        task = retrieval.get_task_class(task_name)
        task_preds = torch.load(input_filepath)[task_name]
        indexes, predictions = task.super_glue_format_preds(task_preds)

        with open(output_filepath, "w") as f:
            for entry in formatted_preds:
                json.dump(entry, f)
                f.write("\n")
        print(task_name, ":", os.path.abspath(output_filepath))


if __name__ == "__main__":
    main()
