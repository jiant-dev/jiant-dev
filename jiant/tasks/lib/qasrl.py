import pandas as pd
import gzip
import json
from dataclasses import dataclass

from jiant.tasks.lib.templates import span_prediction as span_pred_template
from jiant.utils.retokenize import TokenAligner, MosesTokenizer


class QASRLTask(span_pred_template.AbstractSpanPredicationTask):
    def get_train_examples(self):
        return self._create_examples(self.train_path, set_type="train")

    def get_val_examples(self):
        return self._create_examples(self.val_path, set_type="val")

    def get_test_examples(self):
        return self._create_examples(self.test_path, set_type="test")

    def _create_examples(self, file_path, set_type):

        with gzip.open(file_path) as f:
            lines = f.read().splitlines()

        examples = []
        moses_tokenizer = MosesTokenizer()

        for line in lines:
            datum = json.loads(line)
            datum = {
                "sentence_tokens": datum["sentenceTokens"],
                "entries": [
                    {
                        "verb": verb_entry["verbInflectedForms"]["stem"],
                        "verb_idx": verb_idx,
                        "questions": {
                            question: [
                                [
                                    {
                                        "tokens": datum["sentenceTokens"][span[0] : span[1] + 1],
                                        "span": span,
                                    }
                                    for span in answer_judgment["spans"]
                                ]
                                for answer_judgment in q_data["answerJudgments"]
                                if answer_judgment["isValid"]
                            ]
                            for question, q_data in verb_entry["questionLabels"].items()
                        },
                    }
                    for verb_idx, verb_entry in datum["verbEntries"].items()
                ],
            }

            passage_ptb_tokens = datum["sentence_tokens"]
            passage_space_tokens = moses_tokenizer.detokenize_ptb(passage_ptb_tokens).split()
            passage_space_str = " ".join(passage_space_tokens)

            token_aligner = TokenAligner(source=passage_ptb_tokens, target=passage_space_tokens)
            ptb_token_idx_to_space_char_idx = token_aligner.U.dot(token_aligner.C)

            for entry in datum["entries"]:
                for question, answer_list in entry["questions"].items():
                    for answer in answer_list:
                        for answer_span in answer:
                            answer_token_start = answer_span["span"][0]
                            answer_token_end = answer_span["span"][1]

                            nonzero_idxs = (
                                ptb_token_idx_to_space_char_idx[
                                    answer_token_start : answer_token_end + 1
                                ]
                                .sum(axis=0)
                                .nonzero()[0]
                                .tolist()
                            )

                            try:
                                answer_char_span = (nonzero_idxs[0], nonzero_idxs[-1])
                            except Exception:
                                import IPython

                                IPython.embed()

                            examples.append(
                                span_pred_template.Example(
                                    guid="%s-%s" % (set_type, len(examples)),
                                    passage=passage_space_str,
                                    question=question,
                                    answer=passage_space_str[
                                        answer_char_span[0] : answer_char_span[1] + 1
                                    ],
                                    answer_char_span=answer_char_span,
                                )
                            )

        return examples
