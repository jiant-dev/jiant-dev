.. _examples:

=========
Tutorials
=========

Adding a Task
=============

How to Add a Task
-----------------
So you want to use ``jiant`` on a new task? Don't worry, that's easy.

Let's add an imaginary pair classification task called SomeTask with an imaginary dataset called SomeDataset. SomeDataset consists of examples with two sentences (sentence_1 and sentence_2) and a binary label (True and False). To evaluate this task, we will use F1 score. To train this task, we will use mean square error loss.

Add a Task to Jiant's Library of Tasks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Let's add our task to Jiant's task library by adding ``sometask.py`` to Jiant's task library ``jiant/tasks/lib``. We will start by import the modules necessary to describe a task. ``jiant`` contains common task functionality in ``jiant/tasks/lib/templates/shared.py``.::

	import numpy as np
	import torch
	from dataclasses import dataclass
	from typing import List

	from jiant.tasks.core import (
	    BaseExample,
	    BaseTokenizedExample,
	    BaseDataRow,
	    BatchMixin,
	    Task,
	    TaskTypes,
	)
	from jiant.tasks.lib.templates.shared import double_sentence_featurize, labels_to_bimap
	from jiant.utils.python.io import read_json_lines

SomeDataset will be converted to a list of ``Examples``. Here we define a class to represent an instance of the raw dataset. Each example needs to support returning a tokenized version of itself with the ``tokenize(tokenizer)`` method.::

	@dataclass
	class Example(BaseExample):
	    guid: str
	    sentence_1: str
	    sentence_2: str
	    label: str

	    def tokenize(self, tokenizer):
	        return TokenizedExample(
	            guid=self.guid,
	            sentence_1=tokenizer.tokenize(self.sentence_1),
	            sentence_2=tokenizer.tokenize(self.sentence_2),
	            label_id=SomeTask.LABEL_TO_ID[self.label],
	        )

Here we define a class to represent the tokenized version of the ``Example`` above. Generally, this means that sentence_1 and sentence_2 are now ``List``s of tokens instead of strings. Each ``TokenizedExample`` needs to support a ``featurize`` method. When a ``TokenizedExample`` is featurized, it is converted to a ``DataRow`` described below.::

	@dataclass
	class TokenizedExample(BaseTokenizedExample):
	    guid: str
	    sentence_1: List
	    sentence_2: List
	    label_id: int

	    def featurize(self, tokenizer, feat_spec):
	        return double_sentence_featurize(
	            guid=self.guid,
	            input_tokens_a=self.sentence_1,
	            input_tokens_b=self.sentence_2,
	            label_id=self.label_id,
	            tokenizer=tokenizer,
	            feat_spec=feat_spec,
	            data_row_class=DataRow,
	        )

A ``DataRow`` is a single instance of a featurized example.::

	@dataclass
	class DataRow(BaseDataRow):
	    guid: str
	    input_ids: np.ndarray
	    input_mask: np.ndarray
	    segment_ids: np.ndarray
	    label_id: int
	    tokens: list

A ``Batch`` contains DataRows that have been converted to Tensors and are the specified batch size.::

	@dataclass
	class Batch(BatchMixin):
	    input_ids: torch.LongTensor
	    input_mask: torch.LongTensor
	    segment_ids: torch.LongTensor
	    label_id: torch.LongTensor
	    tokens: list

Finally, we add the ``SomeTask`` class. Here we specify the attributes of SomeTask that we defined above. A task must specify the ``Example``, ``TokenizedExample``, ``DataRow``, and ``Batch``. In addition, the task must specify the ``TASK_TYPE``, ``LABELS``, ``LABEL_TO_ID``, and ``ID_TO_LABEL``. Furthermore, the task must implement ``_create_examples(cls, lines, set_type)``, a method to create ``Example``s from SomeDataset. Lastly, the task must implement ``get_train_examples(lines, set_type)``, ``get_val_examples(lines, set_type)``, and ``get_test_examples(lines, set_type)`` to generate a list of ``Examples`` for train, validation, and test sets.::

	class SomeTask(Task):
	    Example = Example
	    TokenizedExample = Example
	    DataRow = DataRow
	    Batch = Batch

	    TASK_TYPE = TaskTypes.CLASSIFICATION
	    LABELS = [False, True]
	    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

	    def get_train_examples(self):
	        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")

	    def get_val_examples(self):
	        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")

	    def get_test_examples(self):
	        return self._create_examples(lines=read_json_lines(self.test_path), set_type="test")

	    @classmethod
	    def _create_examples(cls, lines, set_type):
	        examples = []
	        for (i, line) in enumerate(lines):
	            examples.append(
	                Example(
	                    guid="%s-%s" % (set_type, i),
	                    sentence_1=line["sentence_1"],
	                    sentence_2=line["sentence_2"],
	                    label=line["label"] if set_type != "test" else cls.LABELS[-1],
	                )
	            )
	        return examples


Add Task Evaluation Metric
^^^^^^^^^^^^^^^^^^^^^^^^^^
In ``jiant/tasks/evaluate/core.py``, add SomeTask to the evaluation scheme in the corresponding if-block. Custom evaluation schemes can be added to this file::

	def get_evaluation_scheme_for_task(task) -> BaseEvaluationScheme:
		if isinstance(task, (tasks.SomeTask)):
	    	return AccAndF1EvaluationScheme()

Specify Task in Retrieval Dictionary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In ``jiant/jiant/tasks/retrieval.py``, specify the task name string to task class mapping. Different tasks with the same dataset structure can use the same task class.::

	TASK_DICT = {"some": SomeTask}