import copy

import torch.nn as nn

from dataclasses import dataclass
from typing import Dict, Tuple

import transformers.modeling_bert as modeling_bert
import jiant.utils.torch_utils as torch_utils
import jiant.utils.python.datastructures as datastructures
import jiant.shared.model_resolution as model_resolution
from jiant.proj.main.modeling.primary import JiantModel

DEFAULT_ADAPTER_HIDDEN_ACT = "gelu"
DEFAULT_ADAPTER_SIZE = 64
DEFAULT_ADAPTER_INITIALIZER_RANGE = 0.0002


@dataclass
class AdapterConfig(datastructures.ExtendedDataClassMixin):
    hidden_act: str = DEFAULT_ADAPTER_HIDDEN_ACT
    adapter_size: int = DEFAULT_ADAPTER_SIZE
    adapter_initializer_range: float = DEFAULT_ADAPTER_INITIALIZER_RANGE

    def get_activation(self):
        return modeling_bert.ACT2FN[self.hidden_act] \
            if isinstance(self.hidden_act, str) else self.hidden_act


class Adapter(nn.Module):
    def __init__(self, hidden_size: int, adapter_config: AdapterConfig):
        super(Adapter, self).__init__()
        self.hidden_size = hidden_size
        self.adapter_config = adapter_config

        self.down_project = nn.Linear(
            self.hidden_size,
            self.adapter_config.adapter_size,
        )
        self.activation = adapter_config.get_activation()
        self.up_project = nn.Linear(self.adapter_config.adapter_size, self.hidden_size)
        self.init_weights()

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        return hidden_states + up_projected

    def init_weights(self):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.up_project.bias.data.zero_()


class BertOutputWithAdapters(nn.Module):
    def __init__(self, dense, adapter, layer_norm, dropout):
        super(BertOutputWithAdapters, self).__init__()
        self.dense = dense
        self.adapter = adapter
        self.LayerNorm = layer_norm
        self.dropout = dropout

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    @classmethod
    def from_original(cls, old_module, adapter_config: AdapterConfig):
        assert isinstance(old_module, modeling_bert.BertOutput)
        return cls(
            dense=old_module.dense,
            adapter=Adapter(
                hidden_size=old_module.dense.out_features,
                adapter_config=adapter_config,
            ),
            layer_norm=old_module.LayerNorm,
            dropout=old_module.dropout,
        )


class BertSelfOutputWithAdapters(nn.Module):
    def __init__(self, dense, adapter, layer_norm, dropout):
        super(BertSelfOutputWithAdapters, self).__init__()
        self.dense = dense
        self.adapter = adapter
        self.LayerNorm = layer_norm
        self.dropout = dropout

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    @classmethod
    def from_original(cls, old_module, adapter_config: AdapterConfig):
        assert isinstance(old_module, modeling_bert.BertSelfOutput)
        return cls(
            dense=old_module.dense,
            adapter=Adapter(
                hidden_size=old_module.dense.out_features,
                adapter_config=adapter_config,
            ),
            layer_norm=old_module.LayerNorm,
            dropout=old_module.dropout,
        )


def get_optimized_named_parameters_for_jiant_model_with_adapters(jiant_model: JiantModel):
    """Does a couple things:
    1. Finds the adapter parameters and taskmodel heads (the only params to be optimized)
    2. Sets the other parameters to not require gradients
    """
    set_to_no_grad_list = []
    optimized_named_parameters = []
    for name, param in jiant_model.named_parameters():
        if name.startswith("encoder.") and "adapter" not in name:
            # Do not optimize the shared encoder
            torch_utils.set_requires_grad_single(param, requires_grad=False)
            set_to_no_grad_list.append(name)
        elif ".encoder." in name and "adapter" not in name:
            # Do not optimize the taskmodel encoder weights UNLESS they are adapter modules
            # I believe this strictly speaking isn't necessary because .named_parameters()
            # doesn't return duplicates, but better to be safe.
            torch_utils.set_requires_grad_single(param, requires_grad=False)
            set_to_no_grad_list.append(name)
        else:
            # This should include adapters and taskmodel heads
            optimized_named_parameters.append((name, param))

    return optimized_named_parameters, set_to_no_grad_list


def get_optimized_state_dict_for_jiant_model_with_adapters(jiant_model: JiantModel):
    """Get the state_dict for relevant weights for a JiantModel with adapters

    Basically, the tensors for the adapters and taskmodel heads
    """
    dropped = []
    kept = {}
    for name, tensor in jiant_model.state_dict().items():
        if name.startswith("encoder.") and "adapter" not in name:
            dropped.append(name)
        elif ".encoder." in name and "adapter" not in name:
            # Do not keep the taskmodel encoder weights UNLESS they are adapter modules
            dropped.append(name)
        else:
            # This should include adapters and taskmodel heads
            kept[name] = tensor
    return kept, dropped


def load_state_dict_for_jiant_model_with_adapters(jiant_model: JiantModel, state_dict: Dict):
    """Load state_dict into jiant model, allowing for missing_keys (untrained encoder)

    The checks aren't very rigorous
    """
    mismatched = jiant_model.load_state_dict(state_dict, strict=False)
    assert mismatched.missing_keys
    assert not mismatched.unexpected_keys


def delegate_load_for_shared_adapters(jiant_model: JiantModel, state_dict: Dict, load_mode: str):
    """Different loading methods for shared-adapters"""
    if load_mode == "full":
        load_state_dict_for_jiant_model_with_adapters(
            jiant_model=jiant_model,
            state_dict=state_dict,
        )
    elif load_mode == "adapters_only":
        partial_state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("taskmodels_dict")
        }
        load_state_dict_for_jiant_model_with_adapters(
            jiant_model=jiant_model,
            state_dict=partial_state_dict,
        )
    else:
        raise KeyError(load_mode)


def add_adapters_to_jiant_model_for_each_taskmodel(jiant_model: JiantModel, adapter_config: AdapterConfig):
    """Adds adapters to each taskmodel"""
    adapter_modules_dict = {}
    for taskmodel_name, taskmodel in jiant_model.taskmodels_dict.items():
        encoder_with_adapters, adapter_modules = get_copy_of_encoder_with_adapters(
            encoder=taskmodel.encoder,
            adapter_config=adapter_config,
        )
        taskmodel.encoder = encoder_with_adapters
        adapter_modules_dict[taskmodel_name] = adapter_modules
    return adapter_modules_dict


def add_shared_adapters_to_jiant_model(jiant_model: JiantModel, adapter_config: AdapterConfig):
    """Adds shared adapters to jiant_model"""
    model_architecture = model_resolution.ModelArchitectures.from_encoder(jiant_model.encoder)
    if model_architecture in [model_resolution.ModelArchitectures.BERT,
                              model_resolution.ModelArchitectures.ROBERTA]:
        add_adapters_to_bert_encoder(
            bert_encoder=jiant_model.encoder,
            adapter_config=adapter_config,
        )
    else:
        raise KeyError(model_architecture)


def get_copy_of_encoder_with_adapters(encoder: nn.Module,
                                      adapter_config: AdapterConfig) -> Tuple[nn.Module, Dict]:
    """Resolves module architecture and returns a copy of the encoder with adapters, and a
    dictionary of adapter modules added
    """
    model_architecture = model_resolution.ModelArchitectures.from_encoder(encoder)
    if model_architecture in [model_resolution.ModelArchitectures.BERT,
                              model_resolution.ModelArchitectures.ROBERTA]:
        return get_copy_of_bert_encoder_with_adapters(
            bert_encoder=encoder,
            adapter_config=adapter_config
        )
    else:
        raise KeyError(model_architecture)


def get_copy_of_bert_encoder_with_adapters(bert_encoder: modeling_bert.BertModel,
                                           adapter_config: AdapterConfig) -> Tuple[nn.Module, Dict]:
    """Returns a copy of BertModel with adapters, and a dictionary of adapter modules added

    We're going to make a deepcopy, and then reassign the old parameters
    """
    assert isinstance(bert_encoder, modeling_bert.BertModel)
    new_bert_encoder = copy.deepcopy(bert_encoder)
    adapter_modules = add_adapters_to_bert_encoder(
        bert_encoder=new_bert_encoder,
        adapter_config=adapter_config,
    )
    for name, param in bert_encoder.named_parameters():
        *prefixes, leaf_param_name = name.split(".")
        curr = new_bert_encoder
        for prefix in prefixes:
            curr = getattr(curr, prefix)
        setattr(curr, leaf_param_name, param)
    return new_bert_encoder, adapter_modules


def add_adapters_to_bert_encoder(bert_encoder: modeling_bert.BertModel,
                                 adapter_config: AdapterConfig) -> Dict:
    """Modifies a BertModel in-place, returns dictionary of module names -> adapter modules"""
    assert isinstance(bert_encoder, modeling_bert.BertModel)
    adapter_modules = {}
    for p_name, p_module, c_name, c_module in torch_utils.get_parent_child_module_list(bert_encoder):
        if isinstance(c_module, modeling_bert.BertOutput):
            new_module = BertOutputWithAdapters.from_original(
                old_module=c_module,
                adapter_config=adapter_config,
            )
            setattr(p_module, c_name, new_module)
            adapter_modules[f"{p_name}.{c_name}"] = new_module
        elif isinstance(c_module, modeling_bert.BertSelfOutput):
            new_module = BertSelfOutputWithAdapters.from_original(
                old_module=c_module,
                adapter_config=adapter_config,
            )
            setattr(p_module, c_name, new_module)
            adapter_modules[f"{p_name}.{c_name}"] = new_module
    return adapter_modules
