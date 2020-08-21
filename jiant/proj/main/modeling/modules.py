import torch
import torch.nn as nn
import transformers

import jiant.shared.task_aware_unit as tau


class TransNorm(nn.Module, tau.TaskAwareUnit):
    def __init__(self):
        raise NotImplementedError


def replace_layernorm_with_transnorm(encoder):
    raise NotImplementedError


class MultiAdapter(nn.Module, tau.TaskAwareUnit):
    def __init__(self):
        raise NotImplementedError


class AdapterFusion(nn.Module, tau.TaskAwareUnit):
    def __init__(self):
        raise NotImplementedError


class DoubleEncoder(nn.Module, tau.TaskAwareUnit):
    def __init__(self):
        raise NotImplementedError


class CrossStitchUnit(nn.Module):
    def __init__(self):
        raise NotImplementedError


class SluiceUnit(nn.Module):
    def __init__(self):
        raise NotImplementedError
