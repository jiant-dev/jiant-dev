import pytest

import os
import importlib
from collections import Counter
import numpy as np

from jiant.shared import model_resolution
from jiant.tasks import create_task_from_config_path
from jiant.utils.testing.tokenizer import SimpleSpaceTokenizer

CLASSIFICATION_TASKS = ['mnli', 'sst']


@pytest.mark.parametrize("task_name", CLASSIFICATION_TASKS)
def test_featurization_of_double_input_classification_task(task_name: str):
    fixture_test_examples = importlib.import_module(
        "tests.tasks.lib.resources.fixtures.task_examples." + task_name + "." + task_name + "_examples")

    # Test reading the task-specific toy dataset into examples.
    task = create_task_from_config_path(
        os.path.join(os.path.dirname(__file__), "resources/" + task_name + ".json"), verbose=False
    )
    # Test getting train, val, and test examples. Only the contents of train are checked.
    train_examples = task.get_train_examples()
    val_examples = task.get_val_examples()
    test_examples = task.get_test_examples()
    for train_example_dataclass, raw_example_dict in zip(train_examples, fixture_test_examples.TRAIN_EXAMPLES):
        assert train_example_dataclass.to_dict() == raw_example_dict
    assert val_examples
    assert test_examples

    # Testing conversion of examples into tokenized examples
    # the dummy tokenizer requires a vocab — using a Counter here to find that vocab from the data:
    token_counter = Counter()
    for example in train_examples:
        token_counter.update(example.get_input_a().split())
        # Only check input_b if input_b is supported
        if hasattr(example, 'get_input_b'):
            token_counter.update(example.get_input_b().split())
    token_vocab = list(token_counter.keys())
    tokenizer = SimpleSpaceTokenizer(vocabulary=token_vocab)
    tokenized_examples = [example.tokenize(tokenizer) for example in train_examples]
    for tokenized_example, expected_tokenized_example in zip(
            tokenized_examples, fixture_test_examples.TOKENIZED_TRAIN_EXAMPLES
    ):
        assert tokenized_example.to_dict() == expected_tokenized_example

    # Testing conversion of a tokenized example to a featurized example
    train_example_0_length = len(tokenized_examples[0].get_input_a())
    # Only check input_b if input_b is supported
    if hasattr(tokenized_examples[0], 'get_input_b'):
        train_example_0_length += len(tokenized_examples[0].get_input_b())

    feat_spec = model_resolution.build_featurization_spec(
        model_type="bert-", max_seq_length=train_example_0_length
    )
    featurized_examples = [
        tokenized_example.featurize(tokenizer=tokenizer, feat_spec=feat_spec)
        for tokenized_example in tokenized_examples
    ]
    featurized_example_0_dict = featurized_examples[0].to_dict()
    # not bothering to compare the input_ids because they were made by a dummy tokenizer.
    assert "input_ids" in featurized_example_0_dict
    assert featurized_example_0_dict["guid"] == fixture_test_examples.FEATURIZED_TRAIN_EXAMPLE_0["guid"]

    assert np.all(
            featurized_example_0_dict["input_mask"] == fixture_test_examples.FEATURIZED_TRAIN_EXAMPLE_0["input_mask"]
    )
    assert np.all(
            featurized_example_0_dict["segment_ids"] == fixture_test_examples.FEATURIZED_TRAIN_EXAMPLE_0["segment_ids"]
    )
    assert featurized_example_0_dict["label_id"] == fixture_test_examples.FEATURIZED_TRAIN_EXAMPLE_0["label_id"]
    assert featurized_example_0_dict["tokens"] == fixture_test_examples.FEATURIZED_TRAIN_EXAMPLE_0["tokens"]
