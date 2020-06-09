import argparse
import os
import json
import numpy as np

from jiant.utils.testing.tokenizer import SimpleSpaceTokenizer
from jiant.shared import model_resolution
from jiant.tasks.lib.templates.shared import single_sentence_featurize, double_sentence_featurize
from dataclasses import dataclass
from jiant.tasks.core import BaseDataRow

SET_NAMES = ["train", "val", "test"]


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: int
    tokens: list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a task test")
    parser.add_argument("task", help="name of task")
    parser.add_argument("path", help="filepath for raw task data to be added")
    parser.add_argument("number", type=int, help="number of examples to use in task test")
    parser.add_argument("--input_a", type=str, help="name of input_a")
    parser.add_argument("--input_b", type=str, default=None, help="name of input_a")
    parser.add_argument("--overwrite", help="overwrite existing files", action="store_true")
    parser.add_argument("--task_raw_test_data_out_path", default="data")
    args = parser.parse_args()

    # create task test config
    task_config_path = os.path.join(os.path.dirname(__file__), "..", "tasks", "lib", "resources", args.task + ".json")
    if os.path.exists(task_config_path) or args.overwrite:
        with open(task_config_path, "w") as f:
            test_config = {}
            test_config['task'] = args.task
            test_config['name'] = args.task
            set_path_list = {}
            for set_name in SET_NAMES:
                set_path_list[set_name] = os.path.join(args.task_raw_test_data_out_path, args.task, set_name + ".jsonl")
            test_config['paths'] = set_path_list
            f.write(json.dumps(test_config))

    # create task test data (raw) directory if it does not exist
    task_data_dir = os.path.join(
        os.path.dirname(__file__), "..", "tasks", "lib", "resources", "data", args.task
    )
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

    # create fixture examples file with necessary headers
    task_fixture_path = os.path.join(task_fixture_dir, args.task + "_examples.py")
    if not os.path.exists(task_data_dir):
        os.makedirs(task_data_dir)
    elif args.overwrite:
        if os.path.exists(task_data_dir):
            with open(task_fixture_path, "w") as fixture_file:
                fixture_file.write("import numpy as np\n\n")

    # iterate through set types and write raw data/fixture examples for each set
    for set_name in SET_NAMES:
        # read head of file
        with open(os.path.join(args.path, set_name + ".jsonl")) as task_file:
            head = [next(task_file) for x in range(args.number)]

        # write head to test task data directory
        task_data_file = os.path.join(task_data_dir, set_name + ".jsonl")
        with open(task_data_file, "w") as f:
            for elem in head:
                f.write(elem)

        # read data in JSON format
        task_test_data = [json.loads(example) for example in head]
        for idx, example in enumerate(task_test_data):
            example['guid'] = set_name + "-" + str(idx)

        # write fixture examples to file
        with open(os.path.join(task_fixture_dir, args.task + "_examples.py"), "a") as fixture_file:
            # write raw task data examples to file
            fixture_file.write(
                set_name.upper() + "_EXAMPLES = " + json.dumps(task_test_data, indent=4)
            )
            fixture_file.write("\n\n")

            # write tokenized examples to file
            for example in task_test_data:
                token_vocab = set(example[args.input_a].split())
                if args.input_b:
                    token_vocab.update(set(example[args.input_b].split()))
                tokenizer = SimpleSpaceTokenizer(vocabulary=list(token_vocab))
                example[args.input_a] = tokenizer.tokenize(example[args.input_a])
                if args.input_b:
                    example[input_b] = tokenizer.tokenize(example[args.input_b])
                if set_name != "test":
                    if example["label"].isnumeric():
                        example["label_id"] = int(example["label"])
                example.pop("label", None)

            fixture_file.write(
                "TOKENIZED_"
                + set_name.upper()
                + "_EXAMPLES = "
                + json.dumps(task_test_data, indent=4)
            )
            fixture_file.write("\n\n")

            # write first featurized example to file
            example = task_test_data[0]
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
            if set_name in ["train", "val"]:
                if args.input_b:
                    featurized_example = double_sentence_featurize(
                        guid=set_name + "-0",
                        input_tokens_a=example[args.input_a],
                        input_tokens_b=example[args.input_b],
                        label_id=example["label"],
                        tokenizer=tokenizer,
                        feat_spec=feat_spec,
                        data_row_class=DataRow,
                    )
                else:
                    featurized_example = single_sentence_featurize(
                        guid=set_name + "-0",
                        input_tokens=example[args.input_a],
                        label_id=example["label_id"],
                        tokenizer=tokenizer,
                        feat_spec=feat_spec,
                        data_row_class=DataRow,
                    )
            elif set_name is "test":
                if args.input_b:
                    featurized_example = double_sentence_featurize(
                        guid=set_name + "-0",
                        input_tokens_a=example[args.input_a],
                        input_tokens_b=example[args.input_b],
                        label_id=None,
                        tokenizer=tokenizer,
                        feat_spec=feat_spec,
                        data_row_class=DataRow,
                    )
                else:
                    featurized_example = single_sentence_featurize(
                        guid=set_name + "-0",
                        input_tokens=example[args.input_a],
                        label_id=None,
                        tokenizer=tokenizer,
                        feat_spec=feat_spec,
                        data_row_class=DataRow,
                    )

            fixture_file.write(
                "FEATURIZED_"
                + set_name.upper()
                + "_EXAMPLE_0 = "
                + str(featurized_example.to_dict()).replace("array", "np.array")
            )
            fixture_file.write("\n\n")
