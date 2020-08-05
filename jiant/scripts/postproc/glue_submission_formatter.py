"""Translate raw prediction files for GLUE tasks into format expected by GLUE leaderboard.
This script translates raw prediction files for GLUE tasks into the tsv files required
by the GLUE leaderboard. See https://super.gluebenchmark.com/faq for leaderboard info.
"""
import os
import csv
import argparse
import torch

from jiant.tasks import retrieval
from jiant.tasks.constants import GLUE_TASKS

# this maps the GLUE tasks to the filenames expected by the GLUE benchmark
# (https://gluebenchmark.com/)
formatted_pred_output_filenames = {
    "cola": "CoLA.tsv",
    "sst": "SST-2.tsv",
    "mrpc": "MRPC.tsv",
    "stsb": "STS-B.tsv",
    "mnli": "MNLI-m.tsv",
    "mnli_mismatched": "MNLI-mm.tsv",
    "qnli": "QNLI.tsv",
    "qqp": "QQP.tsv",
    "rte": "RTE.tsv",
    "wnli": "WNLI.tsv",
    "glue_diagnostics": "AX.tsv",
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate output files for GLUE benchmark submission"
    )
    parser.add_argument(
        "--input_base_path",
        required=True,
        help="base input path of GLUE tasks (contains task folders)",
    )
    parser.add_argument("--output_path", required=True, help="output path for formatted files")

    args = parser.parse_args()

    # for task_name, input_filepath in raw_pred_input_filepaths.items():
    for task_name in GLUE_TASKS:
        input_filepath = os.path.join(args.input_base_path, task_name, "test_preds.p")
        output_filepath = os.path.join(args.output_path, formatted_pred_output_filenames[task_name])

        task = retrieval.get_task_class(task_name)
        task_preds = torch.load(input_filepath)[task_name]
        indexes, predictions = task.get_glue_preds(task_preds)

        with open(output_filepath, "w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(("index", "prediction"))
            writer.writerows(zip(indexes, predictions))


if __name__ == "__main__":
    main()
