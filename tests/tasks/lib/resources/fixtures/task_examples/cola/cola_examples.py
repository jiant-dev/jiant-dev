import numpy as np

TRAIN_EXAMPLES = [
    {
        "text": "Our friends won't buy this analysis, let alone the next one we propose.",
        "label": "1",
        "guid": "train-0"
    },
    {
        "text": "One more pseudo generalization and I'm giving up.",
        "label": "1",
        "guid": "train-1"
    },
    {
        "text": "One more pseudo generalization or I'm giving up.",
        "label": "1",
        "guid": "train-2"
    },
    {
        "text": "The more we study verbs, the crazier they get.",
        "label": "1",
        "guid": "train-3"
    },
    {
        "text": "Day by day the facts are getting murkier.",
        "label": "1",
        "guid": "train-4"
    }
]

TOKENIZED_TRAIN_EXAMPLES = [
    {
        "text": [
            "Our",
            "friends",
            "won't",
            "buy",
            "this",
            "analysis,",
            "let",
            "alone",
            "the",
            "next",
            "one",
            "we",
            "propose."
        ],
        "guid": "train-0",
        "label_id": 1
    },
    {
        "text": [
            "One",
            "more",
            "pseudo",
            "generalization",
            "and",
            "I'm",
            "giving",
            "up."
        ],
        "guid": "train-1",
        "label_id": 1
    },
    {
        "text": [
            "One",
            "more",
            "pseudo",
            "generalization",
            "or",
            "I'm",
            "giving",
            "up."
        ],
        "guid": "train-2",
        "label_id": 1
    },
    {
        "text": [
            "The",
            "more",
            "we",
            "study",
            "verbs,",
            "the",
            "crazier",
            "they",
            "get."
        ],
        "guid": "train-3",
        "label_id": 1
    },
    {
        "text": [
            "Day",
            "by",
            "day",
            "the",
            "facts",
            "are",
            "getting",
            "murkier."
        ],
        "guid": "train-4",
        "label_id": 1
    }
]

FEATURIZED_TRAIN_EXAMPLE_0 = {'guid': 'train-0', 'input_ids': np.array([ 1,  5, 12, 10,  9, 13, 15,  8,  6,  4,  7, 11,  2]), 'input_mask': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'segment_ids': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'label_id': 1, 'tokens': ['<cls>', 'Our', 'friends', "won't", 'buy', 'this', 'analysis,', 'let', 'alone', 'the', 'next', 'one', '<sep>']}

VAL_EXAMPLES = [
    {
        "text": "The sailors rode the breeze clear of the rocks.",
        "label": "1",
        "guid": "val-0"
    },
    {
        "text": "The weights made the rope stretch over the pulley.",
        "label": "1",
        "guid": "val-1"
    },
    {
        "text": "The mechanical doll wriggled itself loose.",
        "label": "1",
        "guid": "val-2"
    },
    {
        "text": "If you had eaten more, you would want less.",
        "label": "1",
        "guid": "val-3"
    },
    {
        "text": "As you eat the most, you want the least.",
        "label": "0",
        "guid": "val-4"
    }
]

TOKENIZED_VAL_EXAMPLES = [
    {
        "text": [
            "The",
            "sailors",
            "rode",
            "the",
            "breeze",
            "clear",
            "of",
            "the",
            "rocks."
        ],
        "guid": "val-0",
        "label_id": 1
    },
    {
        "text": [
            "The",
            "weights",
            "made",
            "the",
            "rope",
            "stretch",
            "over",
            "the",
            "pulley."
        ],
        "guid": "val-1",
        "label_id": 1
    },
    {
        "text": [
            "The",
            "mechanical",
            "doll",
            "wriggled",
            "itself",
            "loose."
        ],
        "guid": "val-2",
        "label_id": 1
    },
    {
        "text": [
            "If",
            "you",
            "had",
            "eaten",
            "more,",
            "you",
            "would",
            "want",
            "less."
        ],
        "guid": "val-3",
        "label_id": 1
    },
    {
        "text": [
            "As",
            "you",
            "eat",
            "the",
            "most,",
            "you",
            "want",
            "the",
            "least."
        ],
        "guid": "val-4",
        "label_id": 0
    }
]

FEATURIZED_VAL_EXAMPLE_0 = {'guid': 'val-0', 'input_ids': np.array([ 1,  7, 10, 11,  4,  6,  5,  9,  2]), 'input_mask': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]), 'segment_ids': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), 'label_id': 1, 'tokens': ['<cls>', 'The', 'sailors', 'rode', 'the', 'breeze', 'clear', 'of', '<sep>']}

TEST_EXAMPLES = [
    {
        "text": "Bill whistled past the house.",
        "guid": "test-0"
    },
    {
        "text": "The car honked its way down the road.",
        "guid": "test-1"
    },
    {
        "text": "Bill pushed Harry off the sofa.",
        "guid": "test-2"
    },
    {
        "text": "the kittens yawned awake and played.",
        "guid": "test-3"
    },
    {
        "text": "I demand that the more John eats, the more he pay.",
        "guid": "test-4"
    }
]

TOKENIZED_TEST_EXAMPLES = [
    {
        "text": [
            "Bill",
            "whistled",
            "past",
            "the",
            "house."
        ],
        "guid": "test-0"
    },
    {
        "text": [
            "The",
            "car",
            "honked",
            "its",
            "way",
            "down",
            "the",
            "road."
        ],
        "guid": "test-1"
    },
    {
        "text": [
            "Bill",
            "pushed",
            "Harry",
            "off",
            "the",
            "sofa."
        ],
        "guid": "test-2"
    },
    {
        "text": [
            "the",
            "kittens",
            "yawned",
            "awake",
            "and",
            "played."
        ],
        "guid": "test-3"
    },
    {
        "text": [
            "I",
            "demand",
            "that",
            "the",
            "more",
            "John",
            "eats,",
            "the",
            "more",
            "he",
            "pay."
        ],
        "guid": "test-4"
    }
]

FEATURIZED_TEST_EXAMPLE_0 = {'guid': 'test-0', 'input_ids': np.array([1, 8, 7, 5, 2]), 'input_mask': np.array([1, 1, 1, 1, 1]), 'segment_ids': np.array([0, 0, 0, 0, 0]), 'label_id': None, 'tokens': ['<cls>', 'Bill', 'whistled', 'past', '<sep>']}

