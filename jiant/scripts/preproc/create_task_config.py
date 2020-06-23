import argparse
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a task test")
    parser.add_argument("--task", required=True, help="name of task")
    parser.add_argument("--input_dir", required=True, help="directory of raw task data to be added")
    parser.add_argument("--train", default="train.jsonl", help="filename of train data")
    parser.add_argument("--val", default="val.jsonl", help="filename of validation data")
    parser.add_argument("--test", default="test.jsonl", help="filename of test data")
    parser.add_argument("--output_dir", required=True, help="output directory for task confgi")
    parser.add_argument("--tags_to_id", default=None, help="filename of tags_to_id for task") 
    args = parser.parse_args()

    # create task test config
    task_config = os.path.join(args.output_dir, str(args.task) + ".json")

    # create task test config
    with open(task_config, "w") as f:
        test_config = {}
        data_paths = {}
        test_config["task"] = args.task
        test_config["name"] = args.task

        data_paths["train"] = os.path.join(args.input_dir, args.train)
        data_paths["val"] = os.path.join(args.input_dir, args.val)
        data_paths["test"] = os.path.join(args.input_dir, args.test)
        test_config["paths"] = data_paths
        
        if args.tags_to_id:
            test_config["tags_to_id"] = os.path.join(args.input_dir, args.tags_to_id)
        f.write(json.dumps(test_config, indent=4))
