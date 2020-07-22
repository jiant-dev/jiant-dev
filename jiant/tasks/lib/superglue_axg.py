from dataclasses import dataclass

import jiant.tasks.lib.rte as rte


@dataclass
class Example(rte.Example):
    pass


@dataclass
class TokenizedExample(rte.Example):
    pass


@dataclass
class DataRow(rte.DataRow):
    pass


@dataclass
class Batch(rte.Batch):
    pass


class SuperglueWinogenderDiagnosticsTask(rte.RteTask):
    def get_train_examples(self):
        return None

    def get_val_examples(self):
        return None
