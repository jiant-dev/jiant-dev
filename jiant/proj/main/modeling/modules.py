import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

import jiant.shared.task_aware_unit as tau


class TransNorm(nn.Module, tau.TaskAwareUnit):
    def __init__(self, layer_norm_layer, task_names, momentum=0.1):
        super().__init__()
        self.normalized_shape = layer_norm_layer.normalized_shape
        self.eps = layer_norm_layer.eps
        self.elementwise_affine = layer_norm_layer.elementwise_affine
        if self.elementwise_affine:
            self.weight = layer_norm_layer.weight
            self.bias = layer_norm_layer.bias
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        for task_name in task_names:
            self.register_buffer(f"{task_name}_mean", torch.zeros(self.normalized_shape))
            self.register_buffer(f"{task_name}_var", torch.ones(self.normalized_shape))
        self.momentum = momentum

    def forward(self, input):
        if self.training:
            reshaped_input = input.flatten(dim_end=-len(self.normalized_shape) - 1)
            self._buffers[f"{self.tau_task_name}_mean"] *= 1 - self.momentum
            self._buffers[f"{self.tau_task_name}_mean"] += self.momentum * reshaped_input.mean(
                dim=0
            )
            self._buffers[f"{self.tau_task_name}_var"] *= 1 - self.momentum
            self._buffers[f"{self.tau_task_name}_var"] += self.momentum * reshaped_input.var(dim=0)
        layer_norm_output = F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )
        discrepency = torch.stack(
            [
                self._buffers[f"{task_name}_mean"]
                / torch.sqrt(self._buffers[f"{task_name}_var"] + self.eps)
                for task_name in self.task_names
            ],
            dim=0,
        )
        alpha = 1 / (1 + discrepency.max(dim=0)[0] - discrepency.min(dim=0)[0])
        alpha = alpha / torch.sum(alpha) * self.normalized_shape.numel()
        target_shape = [-1] * (len(layer_norm_output.size()) - alpha.size()) + alpha.size().tolist()
        output = layer_norm_output * (1 + alpha).view(*target_shape)
        return output


def replace_layernorm_with_transnorm(encoder, task_names):
    for idx in len(encoder.layer):
        encoder.layer[idx].attention.output.LayerNorm = TransNorm(
            layer_norm_layer=encoder.layer[idx].attention.output.LayerNorm, task_names=task_names
        )
        encoder.layer[idx].output.LayerNorm = TransNorm(
            layer_norm_layer=encoder.layer[idx].output.LayerNorm, task_names=task_names
        )


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * F.sigmoid(input)


class BertOutputWithAdapter(nn.Module, tau.TaskAwareUnit):
    def __init__(
        self, bert_output_layer, task_names, hidden_size, reduction_factor=16, non_linearity="relu",
    ):
        super().__init__()
        self.dense = bert_output_layer.dense
        self.LayerNorm = bert_output_layer.LayerNorm
        self.dropout = bert_output_layer.dropout
        if non_linearity == "relu":
            non_linear_module = nn.ReLU()
            non_linear_module = nn.LeakyReLU()
        elif non_linearity == "swish":
            non_linear_module = Swish()
        self.adapters = nn.ModuleDict(
            {
                task_name: nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // reduction_factor),
                    non_linear_module,
                    nn.Linear(hidden_size // reduction_factor, hidden_size),
                )
                for task_name in task_names
            }
        )
        for a_module in self.adapters.modules():
            if isinstance(a_module, nn.Linear):
                a_module.weight.data.normal_(mean=0.0, std=0.02)
                a_module.bias.data.zero_()

    def __init__(self):
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        adapter_inputs = self.LayerNorm(input_tensor + hidden_states)
        adapter_states = self.adapters[self.tau_task_name](adapter_inputs) + hidden_states
        hidden_states = self.LayerNorm(input_tensor + adapter_states)
        return hidden_states


class BertOutputWithAdapterFusion(BertOutputWithAdapter):
    def __init__(self, bert_output_layer, task_names, hidden_size, reduction_factor, non_linearity):
        super().__init__(
            bert_output_layer, task_names, hidden_size, reduction_factor, non_linearity
        # TODO: fusion layer


    def __init__(self):
        raise NotImplementedError


class CrossStitchUnit(nn.Module):
    def __init__(self):
        raise NotImplementedError


class SluiceUnit(nn.Module):
    def __init__(self):
        raise NotImplementedError
