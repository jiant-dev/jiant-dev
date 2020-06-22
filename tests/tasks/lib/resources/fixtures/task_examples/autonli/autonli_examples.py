import numpy as np

TRAIN_EXAMPLES = [
    {
        "premise": "The award of the title was followed by the award of the Golden Medal.",
        "hypothesis": "The award of the title preceded the award of the Golden Medal.",
        "label": "e",
        "guid": "train-0"
    },
    {
        "premise": "The award of the title was followed by the award of the Golden Medal.",
        "hypothesis": "The award of the title happened concurrently with the award of the Golden Medal.",
        "label": "c",
        "guid": "train-1"
    },
    {
        "premise": "The award of the title was followed by the award of the Golden Medal.",
        "hypothesis": "The award of the title was more prestigious than the award of the Golden Medal.",
        "label": "n",
        "guid": "train-2"
    },
    {
        "premise": "Muisak - the avenger spirit, it surfaces when a person protected by Arutam is murdered.",
        "hypothesis": "Muisak comes out when an Arutam protected person is murdered.",
        "label": "e",
        "guid": "train-3"
    },
    {
        "premise": "Leander and the Catholic bishops immediately introduced a law imposing the conversion of Jews and declared the remains of Aryanism as \"hereticism. \"",
        "hypothesis": "The Catholic bishops endorsed Aryanism.",
        "label": "c",
        "guid": "train-4"
    }
]

TOKENIZED_TRAIN_EXAMPLES = [
    {
        "premise": [
            "The",
            "award",
            "of",
            "the",
            "title",
            "was",
            "followed",
            "by",
            "the",
            "award",
            "of",
            "the",
            "Golden",
            "Medal."
        ],
        "hypothesis": [
            "The",
            "award",
            "of",
            "the",
            "title",
            "preceded",
            "the",
            "award",
            "of",
            "the",
            "Golden",
            "Medal."
        ],
        "guid": "train-0",
        "label_id": 1
    },
    {
        "premise": [
            "The",
            "award",
            "of",
            "the",
            "title",
            "was",
            "followed",
            "by",
            "the",
            "award",
            "of",
            "the",
            "Golden",
            "Medal."
        ],
        "hypothesis": [
            "The",
            "award",
            "of",
            "the",
            "title",
            "happened",
            "concurrently",
            "with",
            "the",
            "award",
            "of",
            "the",
            "Golden",
            "Medal."
        ],
        "guid": "train-1",
        "label_id": 0
    },
    {
        "premise": [
            "The",
            "award",
            "of",
            "the",
            "title",
            "was",
            "followed",
            "by",
            "the",
            "award",
            "of",
            "the",
            "Golden",
            "Medal."
        ],
        "hypothesis": [
            "The",
            "award",
            "of",
            "the",
            "title",
            "was",
            "more",
            "prestigious",
            "than",
            "the",
            "award",
            "of",
            "the",
            "Golden",
            "Medal."
        ],
        "guid": "train-2",
        "label_id": 2
    },
    {
        "premise": [
            "Muisak",
            "-",
            "the",
            "avenger",
            "spirit,",
            "it",
            "surfaces",
            "when",
            "a",
            "person",
            "protected",
            "by",
            "Arutam",
            "is",
            "murdered."
        ],
        "hypothesis": [
            "Muisak",
            "comes",
            "out",
            "when",
            "an",
            "Arutam",
            "protected",
            "person",
            "is",
            "murdered."
        ],
        "guid": "train-3",
        "label_id": 1
    },
    {
        "premise": [
            "Leander",
            "and",
            "the",
            "Catholic",
            "bishops",
            "immediately",
            "introduced",
            "a",
            "law",
            "imposing",
            "the",
            "conversion",
            "of",
            "Jews",
            "and",
            "declared",
            "the",
            "remains",
            "of",
            "Aryanism",
            "as",
            "\"hereticism.",
            "\""
        ],
        "hypothesis": [
            "The",
            "Catholic",
            "bishops",
            "endorsed",
            "Aryanism."
        ],
        "guid": "train-4",
        "label_id": 0
    }
]

FEATURIZED_TRAIN_EXAMPLE_0 = {'guid': 'train-0', 'input_ids': np.array([ 1, 10,  4,  8,  5,  9, 11,  6, 14,  5,  4,  8,  2, 10,  4,  8,  5,
        9, 13,  5,  4,  8,  5,  7, 12,  2]), 'input_mask': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1]), 'segment_ids': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1]), 'label_id': 1, 'tokens': ['<cls>', 'The', 'award', 'of', 'the', 'title', 'was', 'followed', 'by', 'the', 'award', 'of', '<sep>', 'The', 'award', 'of', 'the', 'title', 'preceded', 'the', 'award', 'of', 'the', 'Golden', 'Medal.', '<sep>']}

VAL_EXAMPLES = [
    {
        "premise": "One Israeli was killed and two others, including a six-year-old girl, were seriously wounded in a shooting attack Monday evening near the West Bank city of Ramallah, Israel Radio reported.",
        "hypothesis": "A 6 year old girl was injured in a shooting in Ramallah.",
        "label": "e",
        "guid": "val-0"
    },
    {
        "premise": "In February, the first such center has been opened in Liberia.",
        "hypothesis": "As of February, all such centers are closed in Liberia.",
        "label": "c",
        "guid": "val-1"
    },
    {
        "premise": "Basically, the White House has a communication problem, she told a group that included advice columnist Ann Landers and gossip or feature section writers from The Washington Post, USA Today and the New York Post.",
        "hypothesis": "The source believed the White House was clear in communicating.",
        "label": "c",
        "guid": "val-2"
    },
    {
        "premise": "According to officials from the O.P.C. International Exhibition Co., which arranged for the participation of the Taiwan suppliers, while this is the 16th year that Taiwan is taking part in the event, the floor area occupied by Taiwan exhibitors has also expanded from just 50 square meters in the first year to 1,400 square meters this year.",
        "hypothesis": "The floor area occupied by Taiwan suppliers this year is 1,400 square meters.",
        "label": "e",
        "guid": "val-3"
    },
    {
        "premise": "``It was pretty simple, really,'' Pepperdine coach Paul Westphal said.",
        "hypothesis": "Paul Westphal said that it was actually quite difficult.",
        "label": "c",
        "guid": "val-4"
    }
]

TOKENIZED_VAL_EXAMPLES = [
    {
        "premise": [
            "One",
            "Israeli",
            "was",
            "killed",
            "and",
            "two",
            "others,",
            "including",
            "a",
            "six-year-old",
            "girl,",
            "were",
            "seriously",
            "wounded",
            "in",
            "a",
            "shooting",
            "attack",
            "Monday",
            "evening",
            "near",
            "the",
            "West",
            "Bank",
            "city",
            "of",
            "Ramallah,",
            "Israel",
            "Radio",
            "reported."
        ],
        "hypothesis": [
            "A",
            "6",
            "year",
            "old",
            "girl",
            "was",
            "injured",
            "in",
            "a",
            "shooting",
            "in",
            "Ramallah."
        ],
        "guid": "val-0",
        "label_id": 1
    },
    {
        "premise": [
            "In",
            "February,",
            "the",
            "first",
            "such",
            "center",
            "has",
            "been",
            "opened",
            "in",
            "Liberia."
        ],
        "hypothesis": [
            "As",
            "of",
            "February,",
            "all",
            "such",
            "centers",
            "are",
            "closed",
            "in",
            "Liberia."
        ],
        "guid": "val-1",
        "label_id": 0
    },
    {
        "premise": [
            "Basically,",
            "the",
            "White",
            "House",
            "has",
            "a",
            "communication",
            "problem,",
            "she",
            "told",
            "a",
            "group",
            "that",
            "included",
            "advice",
            "columnist",
            "Ann",
            "Landers",
            "and",
            "gossip",
            "or",
            "feature",
            "section",
            "writers",
            "from",
            "The",
            "Washington",
            "Post,",
            "USA",
            "Today",
            "and",
            "the",
            "New",
            "York",
            "Post."
        ],
        "hypothesis": [
            "The",
            "source",
            "believed",
            "the",
            "White",
            "House",
            "was",
            "clear",
            "in",
            "communicating."
        ],
        "guid": "val-2",
        "label_id": 0
    },
    {
        "premise": [
            "According",
            "to",
            "officials",
            "from",
            "the",
            "O.P.C.",
            "International",
            "Exhibition",
            "Co.,",
            "which",
            "arranged",
            "for",
            "the",
            "participation",
            "of",
            "the",
            "Taiwan",
            "suppliers,",
            "while",
            "this",
            "is",
            "the",
            "16th",
            "year",
            "that",
            "Taiwan",
            "is",
            "taking",
            "part",
            "in",
            "the",
            "event,",
            "the",
            "floor",
            "area",
            "occupied",
            "by",
            "Taiwan",
            "exhibitors",
            "has",
            "also",
            "expanded",
            "from",
            "just",
            "50",
            "square",
            "meters",
            "in",
            "the",
            "first",
            "year",
            "to",
            "1,400",
            "square",
            "meters",
            "this",
            "year."
        ],
        "hypothesis": [
            "The",
            "floor",
            "area",
            "occupied",
            "by",
            "Taiwan",
            "suppliers",
            "this",
            "year",
            "is",
            "1,400",
            "square",
            "meters."
        ],
        "guid": "val-3",
        "label_id": 1
    },
    {
        "premise": [
            "``It",
            "was",
            "pretty",
            "simple,",
            "really,''",
            "Pepperdine",
            "coach",
            "Paul",
            "Westphal",
            "said."
        ],
        "hypothesis": [
            "Paul",
            "Westphal",
            "said",
            "that",
            "it",
            "was",
            "actually",
            "quite",
            "difficult."
        ],
        "guid": "val-4",
        "label_id": 0
    }
]

FEATURIZED_VAL_EXAMPLE_0 = {'guid': 'val-0', 'input_ids': np.array([ 1, 38, 35,  8, 23, 27, 26,  9, 31, 39,  6,  5, 14, 21, 28, 10, 39,
       20, 36, 29, 18, 32, 13,  7, 37, 19, 16, 11,  2, 22, 12, 24, 25, 30,
        8, 34, 10, 39, 20, 10, 17,  2]), 'input_mask': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'segment_ids': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'label_id': 1, 'tokens': ['<cls>', 'One', 'Israeli', 'was', 'killed', 'and', 'two', 'others,', 'including', 'a', 'six-year-old', 'girl,', 'were', 'seriously', 'wounded', 'in', 'a', 'shooting', 'attack', 'Monday', 'evening', 'near', 'the', 'West', 'Bank', 'city', 'of', 'Ramallah,', '<sep>', 'A', '6', 'year', 'old', 'girl', 'was', 'injured', 'in', 'a', 'shooting', 'in', 'Ramallah.', '<sep>']}

TEST_EXAMPLES = [
    {
        "premise": "Muisak - the avenger spirit, it surfaces when a person protected by Arutam is murdered.",
        "hypothesis": "Muisak only surfaces when a person protected by Arutam is kept alive.",
        "label": "c",
        "guid": "test-0"
    },
    {
        "premise": "Muisak - the avenger spirit, it surfaces when a person protected by Arutam is murdered.",
        "hypothesis": "Muisak also surfaces when a person protected by Arutam is stabbed.",
        "label": "n",
        "guid": "test-1"
    },
    {
        "premise": "Leander and the Catholic bishops immediately introduced a law imposing the conversion of Jews and declared the remains of Aryanism as \"hereticism. \"",
        "hypothesis": "Leander is affiliated with the Catholic bishops.",
        "label": "n",
        "guid": "test-2"
    },
    {
        "premise": "Leander and the Catholic bishops immediately introduced a law imposing the conversion of Jews and declared the remains of Aryanism as \"hereticism. \"",
        "hypothesis": "The Catholic Bishops primarily wrote the law.",
        "label": "n",
        "guid": "test-3"
    },
    {
        "premise": "The companies that gave up their domestic demand have turned to foreign markets, the manufacturing industry has become more of a luxury, and Japan's economy has become markedly dependent on foreign markets.",
        "hypothesis": "Japan abolished foreign markets.",
        "label": "c",
        "guid": "test-4"
    }
]

TOKENIZED_TEST_EXAMPLES = [
    {
        "premise": [
            "Muisak",
            "-",
            "the",
            "avenger",
            "spirit,",
            "it",
            "surfaces",
            "when",
            "a",
            "person",
            "protected",
            "by",
            "Arutam",
            "is",
            "murdered."
        ],
        "hypothesis": [
            "Muisak",
            "only",
            "surfaces",
            "when",
            "a",
            "person",
            "protected",
            "by",
            "Arutam",
            "is",
            "kept",
            "alive."
        ],
        "label": "c",
        "guid": "test-0"
    },
    {
        "premise": [
            "Muisak",
            "-",
            "the",
            "avenger",
            "spirit,",
            "it",
            "surfaces",
            "when",
            "a",
            "person",
            "protected",
            "by",
            "Arutam",
            "is",
            "murdered."
        ],
        "hypothesis": [
            "Muisak",
            "also",
            "surfaces",
            "when",
            "a",
            "person",
            "protected",
            "by",
            "Arutam",
            "is",
            "stabbed."
        ],
        "label": "n",
        "guid": "test-1"
    },
    {
        "premise": [
            "Leander",
            "and",
            "the",
            "Catholic",
            "bishops",
            "immediately",
            "introduced",
            "a",
            "law",
            "imposing",
            "the",
            "conversion",
            "of",
            "Jews",
            "and",
            "declared",
            "the",
            "remains",
            "of",
            "Aryanism",
            "as",
            "\"hereticism.",
            "\""
        ],
        "hypothesis": [
            "Leander",
            "is",
            "affiliated",
            "with",
            "the",
            "Catholic",
            "bishops."
        ],
        "label": "n",
        "guid": "test-2"
    },
    {
        "premise": [
            "Leander",
            "and",
            "the",
            "Catholic",
            "bishops",
            "immediately",
            "introduced",
            "a",
            "law",
            "imposing",
            "the",
            "conversion",
            "of",
            "Jews",
            "and",
            "declared",
            "the",
            "remains",
            "of",
            "Aryanism",
            "as",
            "\"hereticism.",
            "\""
        ],
        "hypothesis": [
            "The",
            "Catholic",
            "Bishops",
            "primarily",
            "wrote",
            "the",
            "law."
        ],
        "label": "n",
        "guid": "test-3"
    },
    {
        "premise": [
            "The",
            "companies",
            "that",
            "gave",
            "up",
            "their",
            "domestic",
            "demand",
            "have",
            "turned",
            "to",
            "foreign",
            "markets,",
            "the",
            "manufacturing",
            "industry",
            "has",
            "become",
            "more",
            "of",
            "a",
            "luxury,",
            "and",
            "Japan's",
            "economy",
            "has",
            "become",
            "markedly",
            "dependent",
            "on",
            "foreign",
            "markets."
        ],
        "hypothesis": [
            "Japan",
            "abolished",
            "foreign",
            "markets."
        ],
        "label": "c",
        "guid": "test-4"
    }
]

FEATURIZED_TEST_EXAMPLE_0 = {'guid': 'test-0', 'input_ids': np.array([ 1, 17, 12, 16,  8,  5,  9, 19, 18, 21, 14, 20, 15,  2, 17,  7, 19,
       18, 21, 14, 20, 15, 13,  4, 11, 10,  2]), 'input_mask': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1]), 'segment_ids': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1]), 'label_id': None, 'tokens': ['<cls>', 'Muisak', '-', 'the', 'avenger', 'spirit,', 'it', 'surfaces', 'when', 'a', 'person', 'protected', 'by', '<sep>', 'Muisak', 'only', 'surfaces', 'when', 'a', 'person', 'protected', 'by', 'Arutam', 'is', 'kept', 'alive.', '<sep>']}

