import os
from collections import Counter

import numpy as np

from jiant.tasks import create_task_from_config_path
from jiant.tasks.lib.sst import TokenizedExample
from jiant.utils.testing.tokenizer import SimpleSpaceTokenizer


TRAIN_EXAMPLES = [
    {"guid": "train-0", "text": "hide new secretions from the parental units ", "label": "0"},
    {"guid": "train-1", "text": "contains no wit , only labored gags ", "label": "0"},
    {
        "guid": "train-2",
        "text": "that loves its characters and communicates something rather beautiful about "
        "human nature ",
        "label": "1",
    },
    {
        "guid": "train-3",
        "text": "remains utterly satisfied to remain the same throughout ",
        "label": "0",
    },
    {
        "guid": "train-4",
        "text": "on the worst revenge-of-the-nerds clich\u00e9s the filmmakers could dredge up ",
        "label": "0",
    },
]

TOKENIZED_TRAIN_EXAMPLES = [
    {
        "guid": "train-0",
        "text": ["hide", "new", "secretions", "from", "the", "parental", "units"],
        "label_id": 0,
    },
    {
        "guid": "train-1",
        "text": ["contains", "no", "wit", ",", "only", "labored", "gags"],
        "label_id": 0,
    },
    {
        "guid": "train-2",
        "text": [
            "that",
            "loves",
            "its",
            "characters",
            "and",
            "communicates",
            "something",
            "rather",
            "beautiful",
            "about",
            "human",
            "nature",
        ],
        "label_id": 1,
    },
    {
        "guid": "train-3",
        "text": ["remains", "utterly", "satisfied", "to", "remain", "the", "same", "throughout"],
        "label_id": 0,
    },
    {
        "guid": "train-4",
        "text": [
            "on",
            "the",
            "worst",
            "revenge-of-the-nerds",
            "clich\u00e9s",
            "the",
            "filmmakers",
            "could",
            "dredge",
            "up",
        ],
        "label_id": 0,
    },
]

# set of tokens found in "test_convert_examples_to_tokenized_examples" for use in other tests.
TOKEN_VOCAB = [
    "hide",
    "new",
    "secretions",
    "from",
    "the",
    "parental",
    "units",
    "contains",
    "no",
    "wit",
    ",",
    "only",
    "labored",
    "gags",
    "that",
    "loves",
    "its",
    "characters",
    "and",
    "communicates",
    "something",
    "rather",
    "beautiful",
    "about",
    "human",
    "nature",
    "remains",
    "utterly",
    "satisfied",
    "to",
    "remain",
    "same",
    "throughout",
    "on",
    "worst",
    "revenge-of-the-nerds",
    "clichés",
    "filmmakers",
    "could",
    "dredge",
    "up",
]

FEATURIZED_TRAIN_EXAMPLE_0 = {
    "guid": "train-0",
    "input_ids": np.array([1, 4, 5, 6, 7, 8, 9, 10, 2, 0]),
    "input_mask": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0]),
    "segment_ids": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "label_id": 0,
    "tokens": ["<cls>", "hide", "new", "secretions", "from", "the", "parental", "units", "<sep>"],
}


def test_read_from_file_and_create_examples():
    task = create_task_from_config_path(
        os.path.join(os.path.dirname(__file__), "resources/sst.json"), verbose=True
    )
    examples = task.get_train_examples()
    for example_dataclass, raw_example_dict in zip(examples, TRAIN_EXAMPLES):
        assert example_dataclass.to_dict() == raw_example_dict


def test_convert_examples_to_tokenized_examples():
    task = create_task_from_config_path(
        os.path.join(os.path.dirname(__file__), "resources/sst.json"), verbose=True
    )
    # create examples from hard-coded TRAIN_EXAMPLES to make this test independent of the file read.
    examples = task._create_examples(TRAIN_EXAMPLES, "train")
    # the dummy tokenizer requires a vocab — using a Counter here to find that vocab from the data:
    token_counter = Counter()
    for example in examples:
        token_counter.update(example.text.split())
    token_vocab = list(token_counter.keys())  # result hard-coded into TOKEN_VOCAB for other tests.
    tokenizer = SimpleSpaceTokenizer(vocabulary=token_vocab)
    tokenized_examples = [example.tokenize(tokenizer) for example in examples]
    for tokenized_example, expected_tokenized_example in zip(
        tokenized_examples, TOKENIZED_TRAIN_EXAMPLES
    ):
        assert tokenized_example.to_dict() == expected_tokenized_example


def test_convert_tokenized_example_to_featurized_example():
    # create example from TOKENIZED_TRAIN_EXAMPLES for test isolation from reading/tokenizing steps.
    tokenized_examples = [TokenizedExample.from_dict(ex) for ex in TOKENIZED_TRAIN_EXAMPLES]
    tokenizer = SimpleSpaceTokenizer(vocabulary=TOKEN_VOCAB)
    feat_spec = tokenizer.get_feat_spec(max_seq_length=10)
    featurized_examples = [
        tokenized_example.featurize(tokenizer=tokenizer, feat_spec=feat_spec)
        for tokenized_example in tokenized_examples
    ]
    featurized_example_0_dict = featurized_examples[0].to_dict()
    # not bothering to compare the input_ids because they were made by a dummy tokenizer.
    assert "input_ids" in featurized_example_0_dict
    assert featurized_example_0_dict["guid"] == FEATURIZED_TRAIN_EXAMPLE_0["guid"]
    assert (
        featurized_example_0_dict["input_mask"] == FEATURIZED_TRAIN_EXAMPLE_0["input_mask"]
    ).all()
    assert (
        featurized_example_0_dict["segment_ids"] == FEATURIZED_TRAIN_EXAMPLE_0["segment_ids"]
    ).all()
    assert featurized_example_0_dict["label_id"] == FEATURIZED_TRAIN_EXAMPLE_0["label_id"]
    assert featurized_example_0_dict["tokens"] == FEATURIZED_TRAIN_EXAMPLE_0["tokens"]
