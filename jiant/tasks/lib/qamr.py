import pandas as pd
from dataclasses import dataclass

from jiant.tasks.lib.templates import span_prediction as span_pred_template
from jiant.utils.python.io import read_json_lines
from jiant.utils.retokenize import TokenAligner, MosesTokenizer


class QAMRTask(span_pred_template.AbstractSpanPredicationTask):
    def get_train_examples(self):
        return self._create_examples(self.train_path, set_type="train")

    def get_val_examples(self):
        return self._create_examples(self.val_path, set_type="val")

    def get_test_examples(self):
        return self._create_examples(self.test_path, set_type="test")

    def _create_examples(self, qa_file_path, set_type):
        wiki_df = pd.read_csv(self.path_dict["wiki_dict"], sep="\t", names=["sent_id", "text"])
        wiki_dict = {row["sent_id"]: row["text"] for _, row in wiki_df.iterrows()}

        data_df = pd.read_csv(
            qa_file_path,
            sep="\t",
            header=None,
            names=[
                "sent_id",
                "target_ids",
                "worker_id",
                "qa_index",
                "qa_word",
                "question",
                "answer",
                "response1",
                "response2",
            ],
        )
        data_df["sent"] = data_df["sent_id"].apply(wiki_dict.get)

        examples = []
        moses_tokenizer = MosesTokenizer()
        for i, row in data_df.iterrows():
            # Answer indices are a space-limited list of numbers.
            # We simply take the min/max of the indices
            answer_idxs = list(map(int, row["answer"].split()))
            ans_token_start, ans_token_end = min(answer_idxs), max(answer_idxs)
            passage_ptb_tokens = row["sent"].split()
            passage_space_tokens = moses_tokenizer.detokenize_ptb(passage_ptb_tokens).split()
            passage_space_str = " ".join(passage_space_tokens)

            token_aligner = TokenAligner(source=passage_ptb_tokens, target=passage_space_tokens)
            ptb_token_idx_to_space_char_idx = token_aligner.U.dot(token_aligner.C)
            answer_char_span = (
                (ptb_token_idx_to_space_char_idx[ans_token_start] > 0).argmax(axis=0),
                ptb_token_idx_to_space_char_idx[ans_token_end].cumsum(axis=0).argmax(axis=0),
            )

            examples.append(
                span_pred_template.Example(
                    guid="%s-%s" % (set_type, i),
                    passage=passage_space_str,
                    question=row["question"],
                    answer=passage_space_str[answer_char_span[0] : answer_char_span[1] + 1],
                    answer_char_span=answer_char_span,
                )
            )

        return examples
