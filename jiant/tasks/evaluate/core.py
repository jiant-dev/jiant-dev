import collections
import json
import re
import string

from dataclasses import dataclass

import numpy as np
import pandas as pd
import seqeval.metrics as seqeval_metrics
from sklearn.metrics import f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List

import jiant.shared.model_resolution as model_resolution
import jiant.tasks as tasks
import jiant.tasks.lib.templates.squad_style.core as squad_style
import jiant.tasks.lib.templates.squad_style.utils as squad_style_utils
import jiant.tasks.lib.mlqa as mlqa_lib
from jiant.utils.python.datastructures import ExtendedDataClassMixin
from jiant.utils.python.io import read_json


@dataclass
class Metrics(ExtendedDataClassMixin):
    major: float
    minor: Dict


class BaseEvaluation:
    pass


class BaseAccumulator:
    def update(self, batch_logits, batch_loss, batch, batch_metadata):
        raise NotImplementedError()

    def get_accumulated(self):
        raise NotImplementedError()


class BaseEvaluationScheme:
    def get_accumulator(self) -> BaseAccumulator:
        raise NotImplementedError()

    def get_labels_from_cache_and_examples(self, task, cache, examples):
        # Depending on the task, labels may be more easily extracted from
        #   a cache or raw examples.
        # Provide the EvaluationScheme with either, but delegate to another function
        #   using only one.
        raise NotImplementedError()

    def get_preds_from_accumulator(self, task, accumulator):
        raise NotImplementedError()

    def compute_metrics_from_accumulator(
        self, task, accumulator: BaseAccumulator, tokenizer, labels
    ) -> Metrics:
        raise NotImplementedError()


class ConcatenateLogitsAccumulator(BaseAccumulator):
    def __init__(self):
        self.logits_list = []

    def update(self, batch_logits, batch_loss, batch, batch_metadata):
        self.logits_list.append(batch_logits)

    def get_accumulated(self):
        all_logits = np.concatenate(self.logits_list)
        return all_logits


class ConcatenateLossAccumulator(BaseAccumulator):
    def __init__(self):
        self.loss_list = []

    def update(self, batch_logits, batch_loss, batch, batch_metadata):
        self.loss_list.append(batch_loss)

    def get_accumulated(self):
        all_loss = np.array(self.loss_list)
        return all_loss


class BaseLogitsEvaluationScheme(BaseEvaluationScheme):
    def get_accumulator(self):
        return ConcatenateLogitsAccumulator()

    def get_labels_from_cache_and_examples(self, task, cache, examples):
        return get_label_ids_from_cache(cache=cache)

    def get_preds_from_accumulator(self, task, accumulator):
        raise NotImplementedError()

    def compute_metrics_from_accumulator(
        self, task, accumulator: ConcatenateLogitsAccumulator, tokenizer, labels: list
    ) -> Metrics:
        preds = self.get_preds_from_accumulator(task=task, accumulator=accumulator)
        return self.compute_metrics_from_preds_and_labels(preds=preds, labels=labels)

    def compute_metrics_from_preds_and_labels(self, preds, labels):
        raise NotImplementedError()


class SimpleAccuracyEvaluationScheme(BaseLogitsEvaluationScheme):
    @classmethod
    def get_preds_from_accumulator(cls, task, accumulator):
        logits = accumulator.get_accumulated()
        return np.argmax(logits, axis=1)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        # noinspection PyUnresolvedReferences
        acc = float((preds == labels).mean())
        return Metrics(major=acc, minor={"acc": acc})


class MultiLabelAccAndF1EvaluationScheme(BaseLogitsEvaluationScheme):
    def get_labels_from_cache_and_examples(self, task, cache, examples):
        return get_multi_label_ids_from_cache(cache=cache)

    def get_preds_from_accumulator(self, task, accumulator):
        logits = accumulator.get_accumulated()
        return (logits > 0.5).astype(int)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        # noinspection PyUnresolvedReferences
        acc = float((preds == labels).mean())
        labels = np.array(labels)
        minor = {
            "acc": acc,
            "f1_micro": f1_score(y_true=labels, y_pred=preds, average="micro"),
            "acc_and_f1_micro": (acc + f1_score(y_true=labels, y_pred=preds, average="micro")) / 2,
        }
        return Metrics(major=minor["acc_and_f1_micro"], minor=minor)


class AccAndF1EvaluationScheme(BaseLogitsEvaluationScheme):
    def get_preds_from_accumulator(self, task, accumulator):
        logits = accumulator.get_accumulated()
        return np.argmax(logits, axis=1)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        # noinspection PyUnresolvedReferences
        acc = float((preds == labels).mean())
        labels = np.array(labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        minor = {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }
        return Metrics(major=minor["acc_and_f1"], minor=minor)


class MCCEvaluationScheme(BaseLogitsEvaluationScheme):
    def get_preds_from_accumulator(self, task, accumulator):
        logits = accumulator.get_accumulated()
        return np.argmax(logits, axis=1)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        mcc = matthews_corrcoef(labels, preds)
        return Metrics(major=mcc, minor={"mcc": mcc})


class PearsonAndSpearmanEvaluationScheme(BaseLogitsEvaluationScheme):
    def get_labels_from_cache_and_examples(self, task, cache, examples):
        return get_label_vals_from_cache(cache=cache)

    def get_preds_from_accumulator(self, task, accumulator):
        logits = accumulator.get_accumulated()
        return np.squeeze(logits, axis=-1)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        pearson_corr = float(pearsonr(preds, labels)[0])
        spearman_corr = float(spearmanr(preds, labels)[0])
        minor = {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }
        return Metrics(major=minor["corr"], minor=minor)


class MultipleChoiceAccuracyEvaluationScheme(BaseLogitsEvaluationScheme):
    def get_accumulator(self):
        return ConcatenateLogitsAccumulator()

    @classmethod
    def get_labels_from_examples(cls, task, examples):
        return get_multiple_choice_label_ids_from_examples(task=task, examples=examples)

    def get_labels_from_cache_and_examples(self, task, cache, examples):
        return get_multiple_choice_labels_from_cache(cache=cache)

    def get_preds_from_accumulator(self, task, accumulator):
        return SimpleAccuracyEvaluationScheme.get_preds_from_accumulator(
            task=task, accumulator=accumulator,
        )

    def compute_metrics_from_preds_and_labels(self, preds, labels):
        return SimpleAccuracyEvaluationScheme.compute_metrics_from_preds_and_labels(
            preds=preds, labels=labels
        )


class CommitmentBankEvaluationScheme(BaseLogitsEvaluationScheme):
    def get_preds_from_accumulator(self, task, accumulator):
        logits = accumulator.get_accumulated()
        return np.argmax(logits, axis=1)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        # noinspection PyUnresolvedReferences
        acc = float((preds == labels).mean())
        labels = np.array(labels)
        f11 = f1_score(y_true=labels == 0, y_pred=preds == 0)
        f12 = f1_score(y_true=labels == 1, y_pred=preds == 1)
        f13 = f1_score(y_true=labels == 2, y_pred=preds == 2)
        avg_f1 = mean(f11, f12, f13)
        return Metrics(
            major=mean(acc, avg_f1),
            minor={"acc": acc, "avg_f1": avg_f1, "f11": f11, "f12": f12, "f13": f13},
        )


class MultiRCEvaluationScheme(BaseEvaluationScheme):
    def get_accumulator(self):
        return ConcatenateLogitsAccumulator()

    @classmethod
    def get_labels_from_examples(cls, task, examples):
        label_values = get_label_ids(examples=examples, task=task)
        question_ids = np.array([example.question_id for example in examples])
        assert len(label_values) == len(question_ids)
        return [
            {"label_values": lab, "question_ids": qid}
            for lab, qid in zip(label_values, question_ids)
        ]

    @classmethod
    def get_labels_from_cache(cls, cache):
        label_values = []
        question_ids = []
        for datum in cache.iter_all():
            label_values.append(datum["data_row"].label_id)
            question_ids.append(datum["data_row"].question_id)
        label_values = np.array(label_values)
        question_ids = np.array(question_ids)
        assert len(label_values) == len(question_ids)
        return [
            {"label_values": lab, "question_ids": qid}
            for lab, qid in zip(label_values, question_ids)
        ]

    def get_labels_from_cache_and_examples(self, task, cache, examples):
        return self.get_labels_from_examples(task=task, examples=examples)

    def get_preds_from_accumulator(self, task, accumulator):
        raise NotImplementedError()

    def compute_metrics_from_accumulator(
        self, task, accumulator: ConcatenateLogitsAccumulator, tokenizer, labels: list
    ) -> Metrics:
        preds = self.get_preds_from_accumulator(task=task, accumulator=accumulator)
        return self.compute_metrics_from_preds_and_labels(preds=preds, labels=labels,)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        df = pd.DataFrame(labels)
        assert "label_values" in df.columns
        assert "question_ids" in df.columns
        df["preds"] = preds
        # noinspection PyUnresolvedReferences
        exact_match = (
            df.groupby("question_ids")
            .apply(lambda _: (_["preds"] == _["label_values"]).all())
            .mean()
        )
        exact_match = float(exact_match)
        f1 = f1_score(y_true=df["label_values"], y_pred=df["preds"])
        return Metrics(major=mean(exact_match, f1), minor={"em": exact_match, "f1": f1},)


@dataclass
class RecordLabelData:
    passage_idx: int
    question_idx: int
    entity_str: str
    answers_dict: Dict[str, str]


class ReCordEvaluationScheme(BaseEvaluationScheme):
    def get_accumulator(self):
        return ConcatenateLogitsAccumulator()

    @classmethod
    def get_labels_from_examples(cls, examples):
        return [
            RecordLabelData(
                passage_idx=example.passage_idx,
                question_idx=example.question_idx,
                entity_str=example.passage_idx,
                answers_dict=example.answers_dict,
            )
            for example in examples
        ]

    @classmethod
    def get_labels_from_cache_and_examples(cls, task, cache, examples):
        return cls.get_labels_from_examples(examples=examples)

    def get_preds_from_accumulator(self, task, accumulator):
        # TODO: Revisit ReCord scoring  (Issue #51)
        raise NotImplementedError("Currently need labels ('examples') to compute preds. Refactor.")

    def compute_metrics_from_accumulator(
        self, task, accumulator: ConcatenateLogitsAccumulator, tokenizer, labels: list
    ) -> Metrics:
        logits = accumulator.get_accumulated()
        predictions_dict, metrics = self.compute_preds_and_metrics_from_logits_and_record_labels(
            logits=logits, examples=labels,
        )
        return metrics

    @classmethod
    def compute_preds_and_metrics_from_logits_and_record_labels(
        cls, logits, examples: List[RecordLabelData]
    ):
        psg_qns_idx_dict = {}
        for i, example in examples:
            psq_qns_idx = example.passage_idx, example.question_idx
            if psq_qns_idx not in psg_qns_idx_dict:
                psg_qns_idx_dict[psq_qns_idx] = []
            psg_qns_idx_dict[psq_qns_idx].append(i)

        f1_ls = []
        em_ls = []

        predictions_dict = {}
        for psq_qns_idx, example_indices in psg_qns_idx_dict:
            # answer_dict should be same across all examples with the same psq_qns_idx
            relevant_examples = [examples[i] for i in example_indices]
            golds = list(relevant_examples[0].answers_dict.values())
            psg_qns_logits = logits[example_indices]
            psg_qns_pred = int(np.argmax(psg_qns_logits[:, 1]))  # Take argmax over positive preds
            pred_ans = relevant_examples[psg_qns_pred].entity_str

            # F1
            f1 = cls.metric_max_over_ground_truths(cls.f1_score, pred_ans, golds)
            f1_ls.append(f1)

            # EM
            em = cls.metric_max_over_ground_truths(cls.exact_match_score, pred_ans, golds)
            em_ls.append(em)
            predictions_dict[psq_qns_idx] = psg_qns_pred

        em = sum(em_ls) / len(em_ls)
        f1 = sum(f1_ls) / len(f1_ls)
        minor = {
            "em": em,
            "f1": f1,
            "f1_em": (f1 + em) / 2,
        }
        metrics = Metrics(major=minor["f1_em"], minor=minor,)
        return predictions_dict, metrics

    @classmethod
    def normalize_answer(cls, s):
        """Lower text and remove punctuation, articles and extra whitespace.
        From official ReCoRD eval script
        """

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def f1_score(cls, prediction, ground_truth):
        """Compute normalized token level F1
        From official ReCoRD eval script
        """
        prediction_tokens = cls.normalize_answer(prediction).split()
        ground_truth_tokens = cls.normalize_answer(ground_truth).split()
        common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @classmethod
    def exact_match_score(cls, prediction, ground_truth):
        """Compute normalized exact match
        From official ReCoRD eval script
        """
        return cls.normalize_answer(prediction) == cls.normalize_answer(ground_truth)

    @classmethod
    def metric_max_over_ground_truths(cls, metric_fn, prediction, ground_truths):
        """Compute max metric between prediction and each ground truth.
        From official ReCoRD eval script
        """
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)


class CCGEvaluationScheme(BaseEvaluationScheme):
    def get_accumulator(self):
        return ConcatenateLogitsAccumulator()

    @classmethod
    def get_label_ids_from_cache(cls, cache):
        return [
            {"label_ids": datum["data_row"].label_ids, "label_mask": datum["data_row"].label_mask}
            for datum in cache.iter_all()
        ]

    @classmethod
    def get_labels_from_cache_and_examples(cls, task, cache, examples):
        return cls.get_label_ids_from_cache(cache=cache)

    def get_preds_from_accumulator(self, task, accumulator):
        logits = accumulator.get_accumulated()
        return np.argmax(logits, axis=-1)

    def compute_metrics_from_accumulator(
        self, task, accumulator: ConcatenateLogitsAccumulator, tokenizer, labels: list
    ) -> Metrics:
        preds = self.get_preds_from_accumulator(task=task, accumulator=accumulator)
        return self.compute_metrics_from_preds_and_labels(preds=preds, labels=labels,)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        label_ids = np.stack([row["label_ids"] for row in labels])
        label_mask = np.stack([row["label_mask"] for row in labels])

        # Account for smart-truncate
        assert (label_mask[:, preds.shape[-1] :] == 0).all()
        label_ids = label_ids[:, : preds.shape[-1]]
        label_mask = label_mask[:, : preds.shape[-1]]

        bool_mask = label_mask.reshape(-1).astype(bool)
        flat_preds = preds.reshape(-1)[bool_mask]
        flat_labels = label_ids.reshape(-1)[bool_mask]
        return cls.compute_metrics_from_flat_preds_and_labels(
            flat_preds=flat_preds, flat_labels=flat_labels,
        )

    @classmethod
    def compute_metrics_from_flat_preds_and_labels(cls, flat_preds, flat_labels):
        return SimpleAccuracyEvaluationScheme.compute_metrics_from_preds_and_labels(
            preds=flat_preds, labels=flat_labels,
        )


class F1TaggingEvaluationScheme(BaseEvaluationScheme):
    def get_accumulator(self):
        return ConcatenateLogitsAccumulator()

    @classmethod
    def get_labels(cls, cache, examples):
        labels = [
            {"pos_list": example.pos_list, "label_mask": datum["data_row"].label_mask}
            for datum, example in zip(cache.iter_all(), examples)
        ]
        for label in labels:
            assert len(label["pos_list"]) == label["label_mask"].sum()
        return labels

    @classmethod
    def get_labels_from_cache_and_examples(cls, task, cache, examples):
        return cls.get_labels(cache=cache, examples=examples)

    def get_preds_from_accumulator(self, task, accumulator):
        logits = accumulator.get_accumulated()
        return np.argmax(logits, axis=-1)

    def compute_metrics_from_accumulator(
        self, task, accumulator: ConcatenateLogitsAccumulator, tokenizer, labels: list
    ) -> Metrics:
        preds = self.get_preds_from_accumulator(task=task, accumulator=accumulator)
        return self.compute_metrics_from_preds_and_labels(task=task, preds=preds, labels=labels,)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, task, preds, labels):
        label_mask = np.stack([row["label_mask"] for row in labels])

        # Account for smart-truncate
        assert (label_mask[:, preds.shape[-1] :] == 0).all()
        label_mask = label_mask[:, : preds.shape[-1]].astype(bool)

        labels_for_eval = [label["pos_list"] for label in labels]
        preds_for_eval = []
        assert len(labels) == preds.shape[0]
        for i in range(len(labels)):
            relevant_preds = preds[i][label_mask[i]]
            relevant_preds_pos = [task.LABEL_BIMAP.b[pos_id] for pos_id in relevant_preds]
            preds_for_eval.append(relevant_preds_pos)

        minor = {
            "precision": seqeval_metrics.precision_score(labels_for_eval, preds_for_eval),
            "recall": seqeval_metrics.recall_score(labels_for_eval, preds_for_eval),
            "f1": seqeval_metrics.f1_score(labels_for_eval, preds_for_eval),
        }
        return Metrics(major=minor["f1"], minor=minor,)


class SQuADEvaluationScheme(BaseEvaluationScheme):
    @classmethod
    def get_accumulator(cls) -> BaseAccumulator:
        return ConcatenateLogitsAccumulator()

    @classmethod
    def get_labels_from_cache(cls, cache):
        return [cls.get_label_from_data_row(datum["data_row"]) for datum in cache.iter_all()]

    @classmethod
    def get_labels_from_cache_and_examples(cls, task, cache, examples):
        return cls.get_labels_from_cache(cache=cache)

    def get_preds_from_accumulator(self, task, accumulator):
        raise NotImplementedError("Currently can't be done without access to dataset")

    def compute_metrics_from_accumulator(
        self, task, accumulator: BaseAccumulator, tokenizer, labels
    ) -> Metrics:
        logits = accumulator.get_accumulated()
        results, predictions = squad_style.compute_predictions_logits_v3(
            data_rows=labels,
            logits=logits,
            n_best_size=task.n_best_size,
            max_answer_length=task.max_answer_length,
            do_lower_case=model_resolution.resolve_is_lower_case(tokenizer),
            version_2_with_negative=task.version_2_with_negative,
            null_score_diff_threshold=task.null_score_diff_threshold,
            tokenizer=tokenizer,
        )
        return Metrics(major=(results["f1"] + results["exact"]) / 2, minor=results,)

    @classmethod
    def get_label_from_data_row(cls, data_row):
        return squad_style.PartialDataRow.from_data_row(data_row)


class XlingQAEvaluationScheme(BaseEvaluationScheme):
    @classmethod
    def get_accumulator(cls) -> BaseAccumulator:
        return ConcatenateLogitsAccumulator()

    @classmethod
    def get_labels_from_cache(cls, cache):
        return [cls.get_label_from_data_row(datum["data_row"]) for datum in cache.iter_all()]

    @classmethod
    def get_labels_from_cache_and_examples(cls, task, cache, examples):
        return cls.get_labels_from_cache(cache=cache)

    def get_preds_from_accumulator(self, task, accumulator):
        raise NotImplementedError("Currently can't be done without access to dataset")

    def compute_metrics_from_accumulator(
        self, task, accumulator: BaseAccumulator, tokenizer, labels
    ) -> Metrics:
        logits = accumulator.get_accumulated()
        assert isinstance(task, (tasks.TyDiQATask, tasks.XquadTask))
        lang = task.language
        results, predictions = squad_style.compute_predictions_logits_v3(
            data_rows=labels,
            logits=logits,
            n_best_size=task.n_best_size,
            max_answer_length=task.max_answer_length,
            do_lower_case=model_resolution.resolve_is_lower_case(tokenizer),
            version_2_with_negative=task.version_2_with_negative,
            null_score_diff_threshold=task.null_score_diff_threshold,
            skip_get_final_text=(lang == "zh"),
            tokenizer=tokenizer,
        )
        return Metrics(major=(results["f1"] + results["exact"]) / 2, minor=results,)

    @classmethod
    def get_label_from_data_row(cls, data_row):
        return squad_style.PartialDataRow.from_data_row(data_row)


class MLQAEvaluationScheme(SQuADEvaluationScheme):
    def get_preds_from_accumulator(self, task, accumulator):
        raise NotImplementedError("Too hard for now, too much handled in one giant lib")

    def compute_metrics_from_accumulator(
        self, task, accumulator: BaseAccumulator, tokenizer, labels
    ) -> Metrics:

        # Todo: Fix val labels cache
        # This is a quick hack
        logits = accumulator.get_accumulated()
        partial_examples = squad_style.data_rows_to_partial_examples(data_rows=labels)
        all_pred_results = squad_style.logits_to_pred_results_list(logits)
        assert task.context_language == task.question_language
        lang = task.context_language
        predictions = squad_style_utils.compute_predictions_logits_v2(
            partial_examples=partial_examples,
            all_results=all_pred_results,
            n_best_size=task.n_best_size,
            max_answer_length=task.max_answer_length,
            do_lower_case=model_resolution.resolve_is_lower_case(tokenizer),
            version_2_with_negative=task.version_2_with_negative,
            null_score_diff_threshold=task.null_score_diff_threshold,
            tokenizer=tokenizer,
            skip_get_final_text=(lang == "zh"),
            verbose=True,
        )
        dataset = read_json(task.val_path)["data"]
        results = mlqa_lib.evaluate(dataset=dataset, predictions=predictions, lang=lang,)
        return Metrics(major=(results["f1"] + results["exact_match"]) / 2, minor=results,)


class MLMEvaluationScheme(BaseEvaluationScheme):
    @classmethod
    def get_accumulator(cls) -> BaseAccumulator:
        return ConcatenateLossAccumulator()

    def get_labels_from_cache_and_examples(self, task, cache, examples):
        # This is a dummy function. There are no external labels.
        return [None]

    def get_preds_from_accumulator(self, task, accumulator):
        raise NotImplementedError("Not possible")

    def compute_metrics_from_accumulator(
        self, task, accumulator: BaseAccumulator, tokenizer, labels
    ) -> Metrics:
        loss_list = accumulator.get_accumulated()
        average_loss = mean(loss_list)
        perplexity = np.exp(average_loss)
        return Metrics(
            # Major = negative perplexity
            major=-perplexity,
            minor={"perplexity": perplexity},
        )


def get_evaluation_scheme_for_task(task) -> BaseEvaluationScheme:
    # TODO: move logic to task?  (Issue #52)
    if isinstance(
        task,
        (
            tasks.AdversarialNliTask,
            tasks.AbductiveNliTask,
            tasks.BoolQTask,
            tasks.CopaTask,
            tasks.MnliTask,
            tasks.PawsXTask,
            tasks.QnliTask,
            tasks.RteTask,
            tasks.SciTailTask,
            tasks.SnliTask,
            tasks.SstTask,
            tasks.WiCTask,
            tasks.WSCTask,
            tasks.XnliTask,
        ),
    ):
        return SimpleAccuracyEvaluationScheme()
    elif isinstance(task, tasks.CCGTask):
        return CCGEvaluationScheme()
    elif isinstance(task, tasks.CommitmentBankTask):
        return CommitmentBankEvaluationScheme()
    elif isinstance(task, tasks.ColaTask):
        return MCCEvaluationScheme()
    elif isinstance(
        task,
        (
            tasks.CommonsenseQATask,
            tasks.CosmosQATask,
            tasks.SWAGTask,
            tasks.HellaSwagTask,
            tasks.SocialIQATask,
        ),
    ):
        return MultipleChoiceAccuracyEvaluationScheme()
    elif isinstance(task, (tasks.MrpcTask, tasks.QqpTask, tasks.ReCoRDTask)):
        return AccAndF1EvaluationScheme()
    elif isinstance(task, tasks.Spr1Task):
        return MultiLabelAccAndF1EvaluationScheme()
    elif isinstance(task, (tasks.SquadTask,)):
        return SQuADEvaluationScheme()
    elif isinstance(task, (tasks.TyDiQATask, tasks.XquadTask,)):
        return XlingQAEvaluationScheme()
    elif isinstance(task, tasks.MlqaTask):
        return MLQAEvaluationScheme()
    elif isinstance(task, tasks.MultiRCTask):
        return MultiRCEvaluationScheme()
    elif isinstance(task, tasks.StsbTask):
        return PearsonAndSpearmanEvaluationScheme()
    elif isinstance(task, (tasks.MLMWikitext103Task, tasks.MLMCrosslingualWikiTask)):
        return MLMEvaluationScheme()
    elif isinstance(task, (tasks.UdposPreprocTask, tasks.PanxPreprocTask,)):
        return F1TaggingEvaluationScheme()
    else:
        raise KeyError(task)


def get_label_ids(task, examples):
    return np.array([task.LABEL_TO_ID[example.label] for example in examples])


def get_label_ids_from_data_row(data_row):
    return data_row.label_ids


def get_multi_label_ids_from_cache(cache):
    return np.array(
        [get_label_ids_from_data_row(data_row=datum["data_row"]) for datum in cache.iter_all()]
    )


def get_label_id_from_data_row(data_row):
    return data_row.label_id


def get_label_ids_from_cache(cache):
    return np.array(
        [get_label_id_from_data_row(data_row=datum["data_row"]) for datum in cache.iter_all()]
    )


def get_label_vals_from_cache(cache):
    return np.array(
        [get_label_val_from_data_row(data_row=datum["data_row"]) for datum in cache.iter_all()]
    )


def get_label_val_from_data_row(data_row):
    return data_row.label


def get_multiple_choice_label_ids_from_examples(task, examples):
    return np.array([task.CHOICE_BIMAP.a[example.label] for example in examples])


def get_multiple_choice_label_id_from_data_row(data_row):
    return data_row.label_id


def get_multiple_choice_labels_from_cache(cache):
    return np.array(
        [
            get_multiple_choice_label_id_from_data_row(data_row=datum["data_row"])
            for datum in cache.iter_all()
        ]
    )


def mean(*args) -> float:
    return float(np.mean(args))


def write_metrics(results, output_path, verbose=True):
    results_to_write = {}
    if "loss" in results:
        results_to_write["loss"] = results["loss"]
    if "metrics" in results:
        results_to_write["metrics"] = results["metrics"].to_dict()
    assert results_to_write
    metrics_str = json.dumps(results_to_write, indent=2)
    if verbose:
        print(metrics_str)
    with open(output_path, "w") as f:
        f.write(metrics_str)
