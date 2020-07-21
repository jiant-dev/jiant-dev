from dataclasses import dataclass

import jiant.tasks.lib.templates.entailment_two_classes as entailment_two_classes


@dataclass
class Example(entailment_two_classes.Example):
    pass


@dataclass
class TokenizedExample(entailment_two_classes.Example):
    pass


@dataclass
class DataRow(entailment_two_classes.DataRow):
    pass


@dataclass
class Batch(entailment_two_classes.Batch):
    pass


class SuperglueWinogenderDiagnosticsTask(entailment_two_classes.EntailmentTwoClassesTask):
    def get_train_examples(self):
        return None

    def get_val_examples(self):
        return None
