from dataclasses import dataclass

import jiant.tasks.lib.mnli as mnli


@dataclass
class Example(mnli.Example):
    pass


@dataclass
class TokenizedExample(mnli.TokenizedExample):
    pass


@dataclass
class DataRow(mnli.DataRow):
    pass


@dataclass
class Batch(mnli.Batch):
    pass


class GlueDiagnosticsTask(mnli.MnliTask):
    def get_train_examples(self):
        return None

    def get_val_examples(self):
        return None
