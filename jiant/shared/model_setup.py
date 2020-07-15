import transformers
import torch

from jiant.ext.radam import RAdam
from jiant.shared.model_resolution import ModelArchitectures, resolve_tokenizer_class


def get_tokenizer(model_type, tokenizer_path):
    """Instantiate a tokenizer for a given model type.

    Args:
        model_type (str): model shortcut name.
        tokenizer_path (str): path to tokenizer directory.

    Returns:
        Tokenizer for the given model type.

    """
    model_arch = ModelArchitectures.from_model_type(model_type)
    tokenizer_class = resolve_tokenizer_class(model_type)
    if model_arch in [ModelArchitectures.BERT]:
        if "-cased" in model_type:
            do_lower_case = False
        elif "-uncased" in model_type:
            do_lower_case = True
        else:
            raise RuntimeError(model_type)
    elif model_arch in [
        ModelArchitectures.XLM,
        ModelArchitectures.ROBERTA,
        ModelArchitectures.XLM_ROBERTA,
    ]:
        do_lower_case = False
    elif model_arch in [ModelArchitectures.ALBERT]:
        do_lower_case = True
    else:
        raise RuntimeError(str(tokenizer_class))
    tokenizer = tokenizer_class.from_pretrained(tokenizer_path, do_lower_case=do_lower_case)
    return tokenizer


class OptimizerScheduler:
    def __init__(self, optimizer, scheduler, args):
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.target = [
            [torch.zeros_like(p.data) for p in p_group["params"]] if p_group["shared"] else []
            for p_group in optimizer.param_groups
        ]
        self.global_sim = args.global_sim
        self.target_momentum = args.target_momentum
        self.source_task = args.source_task
        self.target_task = args.target_task
        self.source_amplifier = args.source_amplifier

    def step(self, task=""):
        optimizer_info= {"gradient_weight": []}
        if task == self.target_task:
            for p_group_idx, _ in enumerate(self.target):
                for p_idx, _ in enumerate(self.target[p_group_idx]):
                    self.target[p_group_idx][p_idx].mul_(self.target_momentum).add_(
                        self.optimizer.param_groups[p_group_idx]["params"][p_idx].grad.data,
                        alpha=1.0 - self.target_momentum,
                    )
        elif task == self.source_task:
            grad_sim = []
            target_norm = []
            grad_norm = []
            for p_group_idx, _ in enumerate(self.target):
                grad_sim.append([])
                for p_idx, _ in enumerate(self.target[p_group_idx]):
                    grad_sim[-1].append(
                        (
                            self.target[p_group_idx][p_idx]
                            * self.optimizer.param_groups[p_group_idx]["params"][p_idx].grad.data
                        ).sum()
                    )
                    if self.global_sim:
                        target_norm.append((self.target[p_group_idx][p_idx] ** 2).sum())
                        grad_norm.append(
                            (
                                self.optimizer.param_groups[p_group_idx]["params"][p_idx].grad.data
                                ** 2
                            ).sum()
                        )
            if self.global_sim:
                global_weight = torch.relu(
                    sum([prod for ls in grad_sim for prod in ls])
                    / (sum(target_norm).sqrt() * sum(grad_norm).sqrt() + 1e-8)
                ).pow(1 / self.source_amplifier)
                optimizer_info["gradient_weight"].append(global_weight.cpu().item())
                
            for p_group_idx, _ in enumerate(self.target):
                for p_idx, _ in enumerate(self.target[p_group_idx]):
                    if self.global_sim:
                        weight = global_weight
                    else:
                        weight = torch.relu(
                            grad_sim[p_group_idx][p_idx]
                            / (
                                self.optimizer.param_groups[p_group_idx]["params"][
                                    p_idx
                                ].grad.data.norm()
                                * self.target[p_group_idx][p_idx].norm()
                                + 1e-8
                            )
                        ).pow(1 / self.source_amplifier)
                        optimizer_info["gradient_weight"].append(weight.cpu().item())
                    self.optimizer.param_groups[p_group_idx]["params"][p_idx].grad.mul_(weight)
        # Scheduler updates first
        self.optimizer.step()
        self.scheduler.step()
        return optimizer_info

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict, strict=True):
        self.optimizer.load_state_dict(state_dict["optimizer"], strict=strict)
        self.scheduler.load_state_dict(state_dict["scheduler"], strict=strict)


def create_optimizer(
    model,
    learning_rate,
    t_total,
    warmup_steps,
    warmup_proportion,
    optimizer_epsilon=1e-8,
    optimizer_type="adam",
    verbose=False,
    args=None,
):
    return create_optimizer_from_params(
        named_parameters=list(model.named_parameters()),
        learning_rate=learning_rate,
        t_total=t_total,
        warmup_steps=warmup_steps,
        warmup_proportion=warmup_proportion,
        optimizer_epsilon=optimizer_epsilon,
        optimizer_type=optimizer_type,
        verbose=verbose,
        args=args,
    )


def create_optimizer_from_params(
    named_parameters,
    learning_rate,
    t_total,
    warmup_steps,
    warmup_proportion,
    optimizer_epsilon=1e-8,
    optimizer_type="adam",
    verbose=False,
    args=None,
):
    # Prepare optimizer
    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "adapter.down_project.weight",
        "adapter.up_project.weight",
        "weighted_sum.weights",
    ]
    if verbose:
        print("No optimizer decay for:")
        for n, p in named_parameters:
            if any(nd in n for nd in no_decay):
                print(f"  {n}")

    used_named_parameters = [
        (n, p) for n, p in named_parameters if p.requires_grad and "weighted_sum.weights" not in n
    ]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in used_named_parameters
                if n.startswith("encoder.e") and (not any(nd in n for nd in no_decay))
            ],
            "weight_decay": 0.01,
            "shared": True,
        },
        {
            "params": [
                p
                for n, p in used_named_parameters
                if (not n.startswith("encoder.e")) and (not any(nd in n for nd in no_decay))
            ],
            "weight_decay": 0.005,
            "shared": False,
        },
        {
            "params": [
                p
                for n, p in used_named_parameters
                if n.startswith("encoder.e") and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "shared": True,
        },
        {
            "params": [
                p
                for n, p in used_named_parameters
                if (not n.startswith("encoder.e")) and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "shared": False,
        },
    ]

    if optimizer_type == "adam":
        if verbose:
            print("Using AdamW")
        optimizer = transformers.AdamW(
            optimizer_grouped_parameters, lr=learning_rate, eps=optimizer_epsilon
        )
    elif optimizer_type == "radam":
        if verbose:
            print("Using RAdam")
        optimizer = RAdam(optimizer_grouped_parameters, lr=learning_rate, eps=optimizer_epsilon)
    else:
        raise KeyError(optimizer_type)

    warmup_steps = resolve_warmup_steps(
        t_total=t_total, warmup_steps=warmup_steps, warmup_proportion=warmup_proportion,
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    optimizer_scheduler = OptimizerScheduler(optimizer=optimizer, scheduler=scheduler, args=args)
    return optimizer_scheduler


def resolve_warmup_steps(t_total, warmup_steps, warmup_proportion):
    if warmup_steps is None and warmup_proportion is None:
        raise RuntimeError()
    elif warmup_steps is not None and warmup_proportion is not None:
        raise RuntimeError()
    elif warmup_steps is None and warmup_proportion is not None:
        return warmup_proportion * t_total
    elif warmup_steps is not None and warmup_proportion is None:
        return warmup_steps
    else:
        raise RuntimeError()


def fp16ize(model, optimizer, fp16_opt_level):
    try:
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        from apex import amp
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
        )
    model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
    return model, optimizer


def parallelize_gpu(model):
    return torch.nn.DataParallel(model)


def parallelize_dist(model, local_rank):
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
    )


def raw_special_model_setup(model, optimizer, fp16, fp16_opt_level, n_gpu, local_rank):
    """Perform setup for special modes (e.g., FP16, DataParallel, and/or DistributedDataParallel.

    Args:
        model (nn.Module): torch model object.
        optimizer: TODO
        fp16 (bool): True to enable FP16 mode.
        fp16_opt_level (str): Apex AMP optimization level default mode identifier.
        n_gpu: number of GPUs.
        local_rank (int): Which GPU the script should use in DistributedDataParallel mode.

    Notes:
        Initialization steps performed in init_cuda_from_args() set n_gpu = 1 when local_rank != -1.

    Returns:
        Model and optimizer with the specified special configuration.

    """
    if fp16:
        model, optimizer = fp16ize(model=model, optimizer=optimizer, fp16_opt_level=fp16_opt_level)
    if n_gpu > 1:
        model = parallelize_gpu(model=model)
    if local_rank != -1:
        model = parallelize_dist(model=model, local_rank=local_rank)
    return model, optimizer


def special_model_setup(
    model_wrapper, optimizer_scheduler, fp16, fp16_opt_level, n_gpu, local_rank
):
    model, optimizer = raw_special_model_setup(
        model=model_wrapper.model,
        optimizer=optimizer_scheduler.optimizer,
        fp16=fp16,
        fp16_opt_level=fp16_opt_level,
        n_gpu=n_gpu,
        local_rank=local_rank,
    )
    model_wrapper.model = model
    optimizer_scheduler.optimizer = optimizer
