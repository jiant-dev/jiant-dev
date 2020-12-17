import transformers
import torch
from copy import deepcopy

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
    def __init__(self, optimizer, scheduler):
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler

    def step(self, skip_scheduler=False):
        self.optimizer.step()
        if not skip_scheduler:
            self.scheduler.step()

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict, strict=True):
        self.optimizer.load_state_dict(state_dict["optimizer"], strict=strict)
        self.scheduler.load_state_dict(state_dict["scheduler"], strict=strict)


class OptimizerSchedulerWithGradOps(OptimizerScheduler):
    def __init__(self, grad_sim_metric, grad_sim_nonlinear, **kwargs):
        super().__init__(**kwargs)
        self.grad_sim_metric = grad_sim_metric
        self.grad_sim_nonlinear = grad_sim_nonlinear
        if grad_sim_nonlinear == "":
            self.nonlinear_fn = lambda x: x
        elif grad_sim_nonlinear.startswith("stepfn"):
            self.threshold = float(self.grad_sim_nonlinear.split("_")[1])
            self.nonlinear_fn = lambda x: (x > self.threshold).float()
        elif grad_sim_nonlinear == "relu":
            self.nonlinear_fn = torch.relu
        elif grad_sim_nonlinear == "sqr":
            self.nonlinear_fn = lambda x: (x * x)

    def get_shared_grad(self, copy=False, get_base=True):
        shared_param_grad = [
            [p.grad for p, is_base in zip(g["params"], g["is_base_encoder"]) if get_base == is_base]
            if g["shared"]
            else []
            for g in self.optimizer.param_groups
        ]
        if copy:
            shared_param_grad = deepcopy(shared_param_grad)
        return shared_param_grad

    def weight_grad(self, grad_sim):
        for g_param, g_sim in zip(self.optimizer.param_groups, grad_sim):
            for p_param, p_sim in zip(g_param["params"], g_sim):
                p_param.grad *= p_sim

    def grad_sim(self, grad_a, grad_b, reduce=True):
        assert self.grad_sim_metric in ["cos", "fisher_cos", "dot_product"]
        if "fisher" in self.grad_sim_metric.split("_"):
            grad_a = [[p ** 2 for p in g] for g in grad_a]
            grad_b = [[p ** 2 for p in g] for g in grad_b]

        grad_sim = [
            [torch.sum(p_a * p_b) for p_a, p_b in zip(g_a, g_b)] for g_a, g_b in zip(grad_a, grad_b)
        ]
        if reduce:
            grad_sim = [[sum([sum(g) for g in grad_sim])]]

        if "cos" in self.grad_sim_metric.split("_"):
            sqr_a = [[(p ** 2).sum() for p in g] for g in grad_a]
            sqr_b = [[(p ** 2).sum() for p in g] for g in grad_b]
            if reduce:
                sqr_a = [[sum([sum(g) for g in sqr_a])]]
                sqr_b = [[sum([sum(g) for g in sqr_b])]]
            grad_sim = [
                [
                    sim / (torch.sqrt(a) * torch.sqrt(b) + 1e-10)
                    for sim, a, b in zip(g_sim, g_a, g_b)
                ]
                for g_sim, g_a, g_b in zip(grad_sim, sqr_a, sqr_b)
            ]

        grad_sim = [[self.nonlinear_fn(p) for p in g] for g in grad_sim]

        if reduce:
            return grad_sim[0][0]
        else:
            return grad_sim


class OptimizerSchedulerForReptile(OptimizerScheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def create_optimizer(
    args,
    model,
    learning_rate,
    t_total,
    warmup_steps,
    warmup_proportion,
    optimizer_epsilon=1e-8,
    optimizer_type="adam",
    verbose=False,
):
    return create_optimizer_from_params(
        args=args,
        named_parameters=list(model.named_parameters()),
        learning_rate=learning_rate,
        t_total=t_total,
        warmup_steps=warmup_steps,
        warmup_proportion=warmup_proportion,
        optimizer_epsilon=optimizer_epsilon,
        optimizer_type=optimizer_type,
        verbose=verbose,
    )


def create_optimizer_from_params(
    args,
    named_parameters,
    learning_rate,
    t_total,
    warmup_steps,
    warmup_proportion,
    optimizer_epsilon=1e-8,
    optimizer_type="adam",
    scheduler_type="linear",
    verbose=False,
):
    # Prepare optimizer
    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "adapter.down_project.weight",
        "adapter.up_project.weight",
        "weighted_sum.weights",
        "task_sharing",
        "layer_sharing",
    ]
    if verbose:
        print("No optimizer decay for:")
        for n, p in named_parameters:
            if any(nd in n for nd in no_decay):
                print(f"  {n}")

    used_named_parameters = [(n, p) for n, p in named_parameters if p.requires_grad]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in used_named_parameters
                if (n.startswith("encoder.e") or n.startswith("dds_model.encoder.e"))
                and (not any(nd in n for nd in no_decay))
            ],
            "weight_decay": 0.01,
            "shared": True,
            "is_base_encoder": [
                n.startswith("encoder.e")
                for n, p in used_named_parameters
                if (n.startswith("encoder.e") or n.startswith("dds_model.encoder.e"))
                and (not any(nd in n for nd in no_decay))
            ],
        },
        {
            "params": [
                p
                for n, p in used_named_parameters
                if (not n.startswith("encoder.e") and not n.startswith("dds_model.encoder.e"))
                and (not any(nd in n for nd in no_decay))
            ],
            "weight_decay": 0.005,
            "shared": False,
            "is_base_encoder": None,
        },
        {
            "params": [
                p
                for n, p in used_named_parameters
                if (n.startswith("encoder.e") or n.startswith("dds_model.encoder.e"))
                and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "shared": True,
            "is_base_encoder": [
                n.startswith("encoder.e")
                for n, p in used_named_parameters
                if (n.startswith("encoder.e") or n.startswith("dds_model.encoder.e"))
                and any(nd in n for nd in no_decay)
            ],
        },
        {
            "params": [
                p
                for n, p in used_named_parameters
                if (not n.startswith("encoder.e") and not n.startswith("dds_model.encoder.e"))
                and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "shared": False,
            "is_base_encoder": None,
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

    if scheduler_type == "linear":
        warmup_steps = resolve_warmup_steps(
            t_total=t_total, warmup_steps=warmup_steps, warmup_proportion=warmup_proportion,
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )
    else:
        raise KeyError(scheduler_type)

    if args.runner_type in ["default", "distill"]:
        optimizer_scheduler = OptimizerScheduler(optimizer=optimizer, scheduler=scheduler)
    elif args.runner_type in ["multidds", "dds", "grad_sim"]:
        optimizer_scheduler = OptimizerSchedulerWithGradOps(
            optimizer=optimizer,
            scheduler=scheduler,
            grad_sim_metric=args.grad_sim_metric,
            grad_sim_nonlinear=args.grad_sim_nonlinear,
        )
    elif args.runner_type == "reptile":
        optimizer_scheduler = OptimizerSchedulerForReptile(optimizer=optimizer, scheduler=scheduler)
    else:
        raise KeyError(args.runner_type)
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
