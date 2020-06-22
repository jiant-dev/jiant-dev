import argparse
import os
import json
import numpy as np

from jiant.utils.testing.tokenizer import SimpleSpaceTokenizer
from jiant.shared import model_resolution
from jiant.tasks.lib.templates.shared import (
    single_sentence_featurize,
    double_sentence_featurize,
)
from dataclasses import dataclass
from jiant.tasks.core import BaseDataRow

SET_TYPES = ["train", "val", "test"]


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: int
    tokens: list


def head(file_path, n):
    with open(file_path) as file:
        head = [next(file).strip() for x in range(n)]
    return head


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a task test")
    parser.add_argument("--task", required=True, help="name of task")
    parser.add_argument("--input_dir", required=True, help="directory of raw task data to be added")
    parser.add_argument("--input_a", type=str, default="text", help="name of input_a")
    parser.add_argument("--input_b", type=str, default=None, help="name of input_b")
    parser.add_argument("--input_c", type=str, default=None, help="name of input_c")
    parser.add_argument("--input_d", type=str, default=None, help="name of input_d")
    parser.add_argument("--train", default="train.jsonl", help="filename of train data")
    parser.add_argument("--val", default="val.jsonl", help="filename of validation data")
    parser.add_argument("--test", default="test.jsonl", help="filename of test data")
    parser.add_argument(
        "--train_labels", default=None, help="filename of train labels if separate",
    )
    parser.add_argument(
        "--val_labels", default=None, help="filename of val labels if separate"
    )
    parser.add_argument(
        "--number", type=int, default=5, help="number of examples to use in task test"
    )
    parser.add_argument(
        "--labels_ordered", nargs='+', help="order of labels in LABEL_TO_ID in task"
    )
    args = parser.parse_args()

    # create task test config
    task_config_path = os.path.join(
        os.path.dirname(__file__), "..", "tasks", "lib", "resources", args.task + ".json",
    )
    if os.path.exists(task_config_path):
        with open(task_config_path, "w") as f:
            test_config = {}
            test_config["task"] = args.task
            test_config["name"] = args.task
            set_path_list = {}
            for set_type in SET_TYPES:
                set_path_list[set_type] = os.path.join("data", args.task, set_type + ".jsonl")
            test_config["paths"] = set_path_list
            f.write(json.dumps(test_config, indent=4))

    # create task test data (raw) directory if it does not exist
    task_data_dir = os.path.join("..", "tasks", "lib", "resources", "data", args.task)
    if not os.path.exists(task_data_dir):
        os.makedirs(task_data_dir)

    # create task test fixture directory if it does not exist
    task_fixture_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "tasks",
        "lib",
        "resources",
        "fixtures",
        "task_examples",
        args.task,
    )
    if not os.path.exists(task_fixture_dir):
        os.makedirs(task_fixture_dir)

    # create fixture examples file with necessary headers
    task_fixture_path = os.path.join(task_fixture_dir, args.task + "_examples.py")
    if not os.path.exists(task_data_dir):
        os.makedirs(task_data_dir)
    with open(task_fixture_path, "w") as fixture_file:
        fixture_file.write("import numpy as np\n\n")

    # iterate through set types and write raw data/fixture examples for each set
    for set_type in SET_TYPES:
        label_head = None
        if set_type == "train":
            set_name = args.train
            if args.train_labels:
                label_head = head(
                    os.path.join(args.input_dir, args.train_labels), args.number,
                )
        elif set_type == "val":
            set_name = args.val
            if args.val_labels:
                label_head = head(
                    os.path.join(args.input_dir, args.val_labels), args.number,
                )
        elif set_type == "test":
            set_name = args.test
        else:
            raise RuntimeError(str(set_type) + " not found in SET_TYPES: " + str(SET_TYPES))

        # read head of file
        head_raw = head(os.path.join(args.input_dir, set_name), args.number)
        json_head = []
        for elem in head_raw:
            obj = json.loads(elem)
            for k in list(obj.keys()):
                if k not in [args.input_a, args.input_b, args.input_c, args.input_d, 'label']:
                    obj.pop(k, None)
            json_head.append(obj)

        if label_head is not None:
            for data_row, label_row in zip(json_head, label_head):
                data_row["label"] = label_row

        # write head to test task data directory
        task_data_file = os.path.join(task_data_dir, set_type + ".jsonl")
        with open(task_data_file, "w") as f:
            for elem in json_head:
                json.dump(elem, f)
                f.write("\n")

        # Add guid
        for idx, example in enumerate(json_head):
            example["guid"] = set_type + "-" + str(idx)

        # write fixture examples to file
        with open(os.path.join(task_fixture_dir, args.task + "_examples.py"), "a") as fixture_file:
            # write raw task data examples to file
            fixture_file.write(set_type.upper() + "_EXAMPLES = " + json.dumps(json_head, indent=4))
            fixture_file.write("\n\n")

            # write tokenized examples to file
            for example in json_head:
                token_vocab = set(example[args.input_a].split())
                if args.input_b:
                    token_vocab.update(set(example[args.input_b].split()))
                if args.input_c:
                    token_vocab.update(set(example[args.input_c].split()))
                if args.input_d:
                    token_vocab.update(set(example[args.input_d].split()))

                tokenizer = SimpleSpaceTokenizer(vocabulary=list(token_vocab))
                example[args.input_a] = tokenizer.tokenize(example[args.input_a])
                if args.input_b:
                    example[args.input_b] = tokenizer.tokenize(example[args.input_b])
                if args.input_c:
                    example[args.input_c] = tokenizer.tokenize(example[args.input_c])
                if args.input_d:
                    example[args.input_d] = tokenizer.tokenize(example[args.input_d])
                if set_type != "test":
                    if example["label"].isnumeric():
                        example["label_id"] = int(example["label"])
                    else:
                        example["label_id"] = args.labels_ordered.index(example["label"])
                    example.pop("label", None)

            fixture_file.write(
                "TOKENIZED_" + set_type.upper() + "_EXAMPLES = " + json.dumps(json_head, indent=4)
            )
            fixture_file.write("\n\n")

            # write first featurized example to file
            example = json_head[0]
            example_length = len(example[args.input_a])
            token_vocab = set(example[args.input_a])

            # Only check input_b if input_b is supported
            if args.input_b:
                example_length += len(example[args.input_b])
                token_vocab.update(set(example[args.input_b]))

            tokenizer = SimpleSpaceTokenizer(vocabulary=list(token_vocab))
            feat_spec = model_resolution.build_featurization_spec(
                model_type="bert-", max_seq_length=example_length
            )

            # only provide label field for train and val sets
            if set_type in ["train", "val"]:
                if args.input_b:
                    featurized_example = double_sentence_featurize(
                        guid=set_type + "-0",
                        input_tokens_a=example[args.input_a],
                        input_tokens_b=example[args.input_b],
                        label_id=example["label_id"],
                        tokenizer=tokenizer,
                        feat_spec=feat_spec,
                        data_row_class=DataRow,
                    )
                else:
                    featurized_example = single_sentence_featurize(
                        guid=set_type + "-0",
                        input_tokens=example[args.input_a],
                        label_id=example["label_id"],
                        tokenizer=tokenizer,
                        feat_spec=feat_spec,
                        data_row_class=DataRow,
                    )
            elif set_type is "test":
                if args.input_b:
                    featurized_example = double_sentence_featurize(
                        guid=set_type + "-0",
                        input_tokens_a=example[args.input_a],
                        input_tokens_b=example[args.input_b],
                        label_id=None,
                        tokenizer=tokenizer,
                        feat_spec=feat_spec,
                        data_row_class=DataRow,
                    )
                else:
                    featurized_example = single_sentence_featurize(
                        guid=set_type + "-0",
                        input_tokens=example[args.input_a],
                        label_id=None,
                        tokenizer=tokenizer,
                        feat_spec=feat_spec,
                        data_row_class=DataRow,
                    )

            fixture_file.write(
                "FEATURIZED_"
                + set_type.upper()
                + "_EXAMPLE_0 = "
                + str(featurized_example.to_dict()).replace("array", "np.array")
            )
            fixture_file.write('\n\n')
