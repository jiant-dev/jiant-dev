import os

import torch

import jiant.proj.main.metarunner as jiant_metarunner
import jiant.proj.adapters.modeling as adapters_modeling
import jiant.utils.python.io as py_io
import jiant.utils.torch_utils as torch_utils


class AdaptersMetarunner(jiant_metarunner):

    # This metarunner modifies the original metarunner to support adapter workflows.
    # Specifically, we want to only save the tuned-parameters to best_model.p
    # We, however, do not modify the checkpoint-saving or best_state_dict
    # because the current metarunner API doesn't make it easy to modify that

    def save_model(self):
        """Override to save only optimized parameters"""
        file_name = f"model__{self.train_state.global_steps:09d}"
        torch.save(
            adapters_modeling.get_optimized_state_dict_for_jiant_model_with_adapters(
                torch_utils.get_model_for_saving(self.model)
            ),
            os.path.join(self.output_dir, f"{file_name}.p"),
        )
        py_io.write_json("{}", os.path.join(self.output_dir, f"{file_name}.metadata.json"))
