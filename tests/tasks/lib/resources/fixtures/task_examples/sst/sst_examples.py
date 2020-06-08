import numpy as np

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

FEATURIZED_TRAIN_EXAMPLE_0 = {
    "guid": "train-0",
    "input_ids": np.array([1, 4, 5, 6, 7, 8, 9, 10, 2, 0]),
    "input_mask": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    "segment_ids": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    "label_id": 0,
    "tokens": ["<cls>", "hide", "new", "secretions", "from", "the", "parental", "units", "<sep>"],
}