from typing import Dict
from dataclasses import dataclass
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import jiant.tasks.evaluate as evaluate
import jiant.utils.torch_utils as torch_utils
from jiant.proj.main.components.container_setup import JiantTaskContainer
from jiant.proj.main.modeling.primary import JiantModel, wrap_jiant_forward
from jiant.shared.constants import PHASE
from jiant.shared.runner import (
    complex_backpropagate,
    get_train_dataloader_from_cache,
    get_eval_dataloader_from_cache,
)
from jiant.utils.display import maybe_tqdm
from jiant.utils.python.datastructures import InfiniteYield, ExtendedDataClassMixin

import higher

@dataclass
class RunnerParameters(ExtendedDataClassMixin):
    local_rank: int
    n_gpu: int
    fp16: bool
    max_grad_norm: float


@dataclass
class TrainState(ExtendedDataClassMixin):
    global_steps: int
    task_steps: Dict[str, int]

    @classmethod
    def from_task_name_list(cls, task_name_list):
        return cls(global_steps=0, task_steps={task_name: 0 for task_name in task_name_list})

    def step(self, task_name):
        self.task_steps[task_name] += 1
        self.global_steps += 1


class JiantRunner:
    def __init__(
        self,
        jiant_task_container: JiantTaskContainer,
        jiant_model: JiantModel,
        optimizer_scheduler,
        device,
        rparams: RunnerParameters,
        log_writer,
    ):
        self.jiant_task_container = jiant_task_container
        self.jiant_model = jiant_model
        self.optimizer_scheduler = optimizer_scheduler
        self.device = device
        self.rparams = rparams
        self.log_writer = log_writer

        self.model = self.jiant_model

    def run_train(self):
        for _ in self.run_train_context():
            pass

    def run_train_context(self, verbose=True):
        train_dataloader_dict = self.get_train_dataloader_dict()
        train_state = TrainState.from_task_name_list(
            self.jiant_task_container.task_run_config.train_task_list
        )
        for _ in maybe_tqdm(
            range(self.jiant_task_container.global_train_config.max_steps),
            desc="Training",
            verbose=verbose,
        ):
            self.run_train_step(
                train_dataloader_dict=train_dataloader_dict, train_state=train_state
            )
            yield train_state

    def resume_train_context(self, train_state, verbose=True):
        train_dataloader_dict = self.get_train_dataloader_dict()
        start_position = train_state.global_steps
        for _ in maybe_tqdm(
            range(start_position, self.jiant_task_container.global_train_config.max_steps),
            desc="Training",
            initial=start_position,
            total=self.jiant_task_container.global_train_config.max_steps,
            verbose=verbose,
        ):
            self.run_train_step(
                train_dataloader_dict=train_dataloader_dict, train_state=train_state
            )
            yield train_state

    def run_train_step(self, train_dataloader_dict: dict, train_state: TrainState):
        self.jiant_model.train()
        task_name, task = self.jiant_task_container.task_sampler.pop()
        task_specific_config = self.jiant_task_container.task_specific_configs[task_name]

        loss_val = 0
        for i in range(task_specific_config.gradient_accumulation_steps):
            batch, batch_metadata = train_dataloader_dict[task_name].pop()
            batch = batch.to(self.device)
            model_output = wrap_jiant_forward(
                jiant_model=self.jiant_model, batch=batch, task=task, compute_loss=True,
            )
            loss = self.complex_backpropagate(
                loss=model_output.loss,
                gradient_accumulation_steps=task_specific_config.gradient_accumulation_steps,
            )
            loss_val += loss.item()

        self.optimizer_scheduler.step()
        self.optimizer_scheduler.optimizer.zero_grad()

        train_state.step(task_name=task_name)
        self.log_writer.write_entry(
            "loss_train",
            {
                "task": task_name,
                "task_step": train_state.task_steps[task_name],
                "global_step": train_state.global_steps,
                "loss_val": loss_val / task_specific_config.gradient_accumulation_steps,
            },
        )

    def run_val(self, task_name_list, use_subset=None, return_preds=False, verbose=True):
        evaluate_dict = {}
        val_dataloader_dict = self.get_val_dataloader_dict(
            task_name_list=task_name_list, use_subset=use_subset
        )
        val_labels_dict = self.get_val_labels_dict(
            task_name_list=task_name_list, use_subset=use_subset
        )
        for task_name in task_name_list:
            task = self.jiant_task_container.task_dict[task_name]
            evaluate_dict[task_name] = run_val(
                val_dataloader=val_dataloader_dict[task_name],
                val_labels=val_labels_dict[task_name],
                jiant_model=self.jiant_model,
                task=task,
                device=self.device,
                local_rank=self.rparams.local_rank,
                return_preds=return_preds,
                verbose=verbose,
            )
        return evaluate_dict

    def run_test(self, task_name_list, verbose=True):
        evaluate_dict = {}
        test_dataloader_dict = self.get_test_dataloader_dict()
        for task_name in task_name_list:
            task = self.jiant_task_container.task_dict[task_name]
            evaluate_dict[task_name] = run_test(
                test_dataloader=test_dataloader_dict[task_name],
                jiant_model=self.jiant_model,
                task=task,
                device=self.device,
                local_rank=self.rparams.local_rank,
                verbose=verbose,
            )
        return evaluate_dict

    def get_train_dataloader_dict(self):
        # Not currently supported distributed parallel
        train_dataloader_dict = {}
        for task_name in self.jiant_task_container.task_run_config.train_task_list:
            task = self.jiant_task_container.task_dict[task_name]
            train_cache = self.jiant_task_container.task_cache_dict[task_name]["train"]
            train_batch_size = self.jiant_task_container.task_specific_configs[
                task_name
            ].train_batch_size
            train_dataloader_dict[task_name] = InfiniteYield(
                get_train_dataloader_from_cache(
                    train_cache=train_cache, task=task, train_batch_size=train_batch_size,
                )
            )
        return train_dataloader_dict

    def _get_eval_dataloader_dict(self, phase, task_name_list, use_subset=False):
        val_dataloader_dict = {}
        for task_name in task_name_list:
            task = self.jiant_task_container.task_dict[task_name]
            eval_cache = self.jiant_task_container.task_cache_dict[task_name][phase]
            task_specific_config = self.jiant_task_container.task_specific_configs[task_name]
            val_dataloader_dict[task_name] = get_eval_dataloader_from_cache(
                eval_cache=eval_cache,
                task=task,
                eval_batch_size=task_specific_config.eval_batch_size,
                subset_num=task_specific_config.eval_subset_num if use_subset else None,
            )
        return val_dataloader_dict

    def get_val_dataloader_dict(self, task_name_list, use_subset=False):
        return self._get_eval_dataloader_dict(
            phase="val", task_name_list=task_name_list, use_subset=use_subset,
        )

    def get_val_labels_dict(self, task_name_list, use_subset=False):
        val_labels_dict = {}
        for task_name in task_name_list:
            task_specific_config = self.jiant_task_container.task_specific_configs[task_name]
            val_labels_cache = self.jiant_task_container.task_cache_dict[task_name]["val_labels"]
            val_labels = val_labels_cache.get_all()
            if use_subset:
                val_labels = val_labels[: task_specific_config.eval_subset_num]
            val_labels_dict[task_name] = val_labels
        return val_labels_dict

    def get_test_dataloader_dict(self):
        return self._get_eval_dataloader_dict(
            task_name_list=self.jiant_task_container.task_run_config.test_task_list,
            phase=PHASE.TEST,
        )

    def complex_backpropagate(self, loss, gradient_accumulation_steps):
        return complex_backpropagate(
            loss=loss,
            optimizer=self.optimizer_scheduler.optimizer,
            model=self.jiant_model,
            fp16=self.rparams.fp16,
            n_gpu=self.rparams.n_gpu,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=self.rparams.max_grad_norm,
        )

    def get_runner_state(self):
        # TODO: Add fp16  (Issue #46)
        state = {
            "model": torch_utils.get_model_for_saving(self.jiant_model).state_dict(),
            "optimizer": self.optimizer_scheduler.optimizer.state_dict(),
        }
        return state

    def load_state(self, runner_state):
        torch_utils.get_model_for_saving(self.jiant_model).load_state_dict(runner_state["model"])
        self.optimizer_scheduler.optimizer.load_state_dict(runner_state["optimizer"])


class ReptileRunner(JiantRunner):
    def __init__(self, inner_steps, num_sampled_tasks, **kwarg):
        super().__init__(**kwarg)
        self.inner_steps = inner_steps
        self.num_sampled_tasks = num_sampled_tasks

    def run_train_step(self, train_dataloader_dict: dict, train_state: TrainState):
        self.jiant_model.train()
        task_name, task = self.jiant_task_container.task_sampler.pop()
        task_specific_config = self.jiant_task_container.task_specific_configs[task_name]

        self.optimizer_scheduler.inner_begin()

        loss_val = 0
        for step in range(self.inner_steps):
            for idx_task in range(self.num_sampled_tasks):
                for i in range(task_specific_config.gradient_accumulation_steps):
                    batch, batch_metadata = train_dataloader_dict[task_name].pop()
                    batch = batch.to(self.device)
                    model_output = wrap_jiant_forward(
                        jiant_model=self.jiant_model, batch=batch, task=task, compute_loss=True,
                    )
                    loss = self.complex_backpropagate(
                        loss=model_output.loss,
                        gradient_accumulation_steps=task_specific_config.gradient_accumulation_steps,
                    )
                    loss_val += loss.item()

            self.optimizer_scheduler.inner_step()
        self.optimizer_scheduler.inner_end()

        self.optimizer_scheduler.step()
        self.optimizer_scheduler.optimizer.zero_grad()

        train_state.step(task_name=task_name)
        self.log_writer.write_entry(
            "loss_train",
            {
                "task": task_name,
                "task_step": train_state.task_steps[task_name],
                "global_step": train_state.global_steps,
                "loss_val": loss_val
                / task_specific_config.gradient_accumulation_steps
                / self.num_sampled_tasks
                / self.inner_steps,
            },
        )


class MultiDDSRunner(JiantRunner):
    def __init__(self, sampler_update_freq, target_task, **kwarg):
        super().__init__(**kwarg)
        self.sampler_update_freq = sampler_update_freq
        self.target_task = target_task

    def run_train_step(self, train_dataloader_dict: dict, train_state: TrainState):
        self.jiant_model.train()
        task_name, task = self.jiant_task_container.task_sampler.pop()
        task_specific_config = self.jiant_task_container.task_specific_configs[task_name]

        def run_one_batch(task_name, task):
            task_specific_config = self.jiant_task_container.task_specific_configs[task_name]
            loss_val = 0
            for i in range(task_specific_config.gradient_accumulation_steps):
                batch, batch_metadata = train_dataloader_dict[task_name].pop()
                batch = batch.to(self.device)
                model_output = wrap_jiant_forward(
                    jiant_model=self.jiant_model, batch=batch, task=task, compute_loss=True,
                )
                loss = self.complex_backpropagate(
                    loss=model_output.loss,
                    gradient_accumulation_steps=task_specific_config.gradient_accumulation_steps,
                )
                loss_val += loss.item()
            return loss_val

        loss_val = run_one_batch(task_name, task)
        self.optimizer_scheduler.step()
        self.optimizer_scheduler.optimizer.zero_grad()

        train_state.step(task_name=task_name)
        if train_state.global_steps % self.sampler_update_freq == 0:
            _ = run_one_batch(
                self.target_task,
                self.jiant_task_container.task_sampler.task_dict[self.target_task],
            )
            target_grad = self.optimizer_scheduler.get_shared_grad(copy=True)
            self.optimizer_scheduler.optimizer.zero_grad()

            task_choice = []
            task_grad_sim = []
            for task_idx, (task_name, task) in enumerate(
                self.jiant_task_container.task_sampler.task_dict.items()
            ):
                _ = run_one_batch(task_name, task,)
                auxillary_grad = self.optimizer_scheduler.get_shared_grad(copy=False)
                task_choice.append(task_idx)
                task_grad_sim = self.optimizer_scheduler.grad_sim(
                    target_grad, auxillary_grad, reduce=True
                )
                self.optimizer_scheduler.optimizer.zero_grad()
            self.task_sampler.update_sampler(
                torch.LongTensor(task_choice), torch.cat(task_grad_sim, dim=0)
            )
        self.log_writer.write_entry(
            "loss_train",
            {
                "task": task_name,
                "task_step": train_state.task_steps[task_name],
                "global_step": train_state.global_steps,
                "loss_val": loss_val / task_specific_config.gradient_accumulation_steps,
            },
        )


class DistillationRunner(JiantRunner):
    def __init__(self, t_total, teacher_jiant_model, **kwarg):
        super().__init__(**kwarg)
        self.teacher_jiant_model = teacher_jiant_model
        for p in self.teacher_jiant_model.parameters():
            p.requires_grad = False
        self.t_total = t_total

    def run_train_step(self, train_dataloader_dict: dict, train_state: TrainState):
        self.jiant_model.train()
        task_name, task = self.jiant_task_container.task_sampler.pop()
        task_specific_config = self.jiant_task_container.task_specific_configs[task_name]

        loss_val = 0
        for i in range(task_specific_config.gradient_accumulation_steps):
            batch, batch_metadata = train_dataloader_dict[task_name].pop()
            batch = batch.to(self.device)
            model_output = wrap_jiant_forward(
                jiant_model=self.jiant_model, batch=batch, task=task, compute_loss=True,
            )
            teacher_model_output = wrap_jiant_forward(
                jiant_model=self.teacher_jiant_model, batch=batch, task=task, compute_loss=True,
            )
            teacher_prediction = torch.softmax(teacher_model_output.logits)
            mixing_weight = train_state.global_steps / self.t_total
            from jiant.tasks.core import TaskTypes

            gt_prediction = torch.zeros_like(teacher_prediction)
            if task.TASK_TYPE == TaskTypes.SPAN_COMPARISON_CLASSIFICATION:
                # TODO: run the program, print tensor shapes
                raise NotImplementedError
            elif task.TASK_TYPE == TaskTypes.CLASSIFICATION:
                raise NotImplementedError
            elif task.TASK_TYPE == TaskTypes.SQUAD_STYLE_QA:
                raise NotImplementedError
            elif task.TASK_TYPE == TaskTypes.TAGGING:
                raise NotImplementedError

            mixed_prediction = (
                1 - mixing_weight
            ) * teacher_prediction + mixing_weight * gt_prediction
            loss = self.complex_backpropagate(
                loss=torch.sum(mixed_prediction * torch.logsoftmax(model_output.logits, dim=-1)),
                gradient_accumulation_steps=task_specific_config.gradient_accumulation_steps,
            )
            loss_val += loss.item()

        self.optimizer_scheduler.step()
        self.optimizer_scheduler.optimizer.zero_grad()

        train_state.step(task_name=task_name)
        self.log_writer.write_entry(
            "loss_train",
            {
                "task": task_name,
                "task_step": train_state.task_steps[task_name],
                "global_step": train_state.global_steps,
                "loss_val": loss_val / task_specific_config.gradient_accumulation_steps,
            },
        )


class L2TWWRunner(JiantRunner):
    def __init__(self, teacher_jiant_model, meta_optimizer_scheduler, **kwarg):
        super().__init__(**kwarg)
        self.teacher_jiant_model = teacher_jiant_model
        for p in self.teacher_jiant_model.parameters():
            p.requires_grad = False
        self.what_where_net = self.MetaWhatAndWhere(hidden_size, teacher_num_layers, student_num_layers)
        self.meta_optimizer = meta_optimizer_scheduler

    class WhatNetwork(nn.Module):
        def __init__(self, hidden_size, teacher_num_layers, student_num_layers):
            super().__init__()
            self.hidden_size = hidden_size
            self.teacher_num_layers = teacher_num_layers
            self.student_num_layers = student_num_layers

            # WeightNet (l, hidden* num_target_layers) for all l in source
            # outputs = softmax across hidden for all pairs
            self.what_network_linear = []
            for i in range(teacher_num_layers):
                self.what_network_linear.append(nn.Linear(self.hidden_size, self.student_num_layers * self.hidden_size))

        def forward(self, teacher_states):
            # TODO: compute L_wfm
            outputs = []
            for i in range(len(teacher_states)):
                out = self.what_network_linear[i](teacher_states[i])
                out = out.reshape(self.student_num_layers, self.hidden_size)
                out = F.softmax(out, 1)
                outputs.extend(out)

            return outputs

    class WhereNetwork(nn.Module):
        def __init__(self, hidden_size, teacher_num_layers, student_num_layers):
            super().__init__()
            self.hidden_size = hidden_size
            self.teacher_num_layers = teacher_num_layers
            self.student_num_layers = student_num_layers

            # LossWeightNet (l, num_target_layers) for all l in source
            # outputs => lambdas[0,..., num_pairs]
            self.where_network_linear = []
            for i in range(teacher_num_layers):
                self.where_network_linear.append(nn.Linear(self.hidden_size, self.student_num_layers))

        def forward(self, teacher_states):
            # TODO: compute L_wfm
            outputs = []
            for i in range(self.teacher_num_layers):
                out = F.relu(self.where_network_linear[i](teacher_states[i])).squeeze()
                outputs.extend(out)
            return outputs

    class MetaWhatAndWhere(nn.Module):
        def __init__(self, hidden_size, teacher_num_layers, student_num_layers):
            super().__init__()
            self.what_network = self.WhatNetwork(hidden_size, teacher_num_layers, student_num_layers)
            self.where_network = self.WhereNetwork(hidden_size, teacher_num_layers, student_num_layers)

        def forward(self, teacher_states, student_states):
            # TODO: compute L_wfm
            weights = self.what_network(teacher_states)
            loss_weights = self.where_network(teacher_states)
            matching_loss = self.what_where_net(teacher_states, student_states, weights,
                                                loss_weights)

            matching_loss = 0.0
            for m in range(len(teacher_states)):
                for n in range(len(student_states)):
                    # diff = teacher_states[m] - self.gammas[n](student_states[n])
                    diff = (teacher_states[m] - student_states[n]).pow(2)  # BSZ * Hidden * SEQ_LEN
                    diff = diff.mean(3).mean(2)
                    diff = (diff.mul(weights[m][n]).sum(1) * loss_weights[m][n]).mean(0)
                matching_loss += diff
            return matching_loss

    def run_inner_loop(self,  meta_batches, task, inner_steps=1):
        self.teacher_jiant_model.eval()
        with higher.innerloop_ctx(self.jiant_model, optimizer) as (fmodel, diffopt):
            for batch in meta_batches:
                for inner_idx in range(inner_steps):
                    model_output = wrap_jiant_forward(
                        jiant_model=fmodel, batch=batch, task=task, compute_loss=True,
                    )

                    teacher_model_output = wrap_jiant_forward(
                        jiant_model=self.teacher_jiant_model, batch=batch, task=task, compute_loss=True,
                    )

                    beta = 0.5
                    matching_loss = self.what_where_net(teacher_model_output.other[0], model_output.other[0])
                    total_inner_loss = model_output.loss + matching_loss * beta
                    diffopt.step(total_inner_loss)
                outer_model_output = wrap_jiant_forward(
                        jiant_model=fmodel, batch=batch, task=task, compute_loss=True,
                    )
                outer_loss = outer_model_output.loss
                outer_loss.backward()
                self.meta_optimizer.step()


    def run_train_step(self, train_dataloader_dict: dict, train_state: TrainState):

        # TODO: modify this to
        self.jiant_model.train()
        task_name, task = self.jiant_task_container.task_sampler.pop()
        task_specific_config = self.jiant_task_container.task_specific_configs[task_name]

        loss_val = 0

        meta_batches = []
        for i in range(task_specific_config.gradient_accumulation_steps):
            batch, batch_metadata = train_dataloader_dict[task_name].pop()
            batch = batch.to(self.device)
            meta_batches.append(batch_metadata.to(self.device))
            model_output = wrap_jiant_forward(
                jiant_model=self.jiant_model, batch=batch, task=task, compute_loss=True,
            )
            loss = self.complex_backpropagate(
                loss=model_output.loss,
                gradient_accumulation_steps=task_specific_config.gradient_accumulation_steps,
            )
            loss_val += loss.item()

        self.optimizer_scheduler.step()
        self.optimizer_scheduler.optimizer.zero_grad()

        run_inner_loop(meta_batches, task)

        train_state.step(task_name=task_name)
        self.log_writer.write_entry(
            "loss_train",
            {
                "task": task_name,
                "task_step": train_state.task_steps[task_name],
                "global_step": train_state.global_steps,
                "loss_val": loss_val / task_specific_config.gradient_accumulation_steps,
            },
        )


class CheckpointSaver:
    def __init__(self, metadata, output_dir, multiple_checkpoints):
        self.metadata = metadata
        self.output_dir = output_dir
        self.multiple_checkpoints = multiple_checkpoints
        self.save_count = 0

    def save(self, runner_state: dict, metarunner_state: dict):
        to_save = {
            "runner_state": runner_state,
            "metarunner_state": metarunner_state,
            "metadata": self.metadata,
        }
        if self.multiple_checkpoints:
            save_path = os.path.join(self.output_dir, f"checkpoint_{self.save_count}.p")
            self.save_count += 1
        else:
            save_path = os.path.join(self.output_dir, "checkpoint.p")
        torch_utils.safe_save(to_save, save_path)


def run_val(
    val_dataloader,
    val_labels,
    jiant_model: JiantModel,
    task,
    device,
    local_rank,
    return_preds=False,
    verbose=True,
):
    # Reminder:
    #   val_dataloader contains mostly PyTorch-relevant info
    #   val_labels might contain more details information needed for full evaluation
    if not local_rank == -1:
        return
    jiant_model.eval()
    total_eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    evaluation_scheme = evaluate.get_evaluation_scheme_for_task(task=task)
    eval_accumulator = evaluation_scheme.get_accumulator()

    for step, (batch, batch_metadata) in enumerate(
        maybe_tqdm(val_dataloader, desc=f"Eval ({task.name}, Val)", verbose=verbose)
    ):
        batch = batch.to(device)

        with torch.no_grad():
            model_output = wrap_jiant_forward(
                jiant_model=jiant_model, batch=batch, task=task, compute_loss=True,
            )
        batch_logits = model_output.logits.detach().cpu().numpy()
        batch_loss = model_output.loss.mean().item()
        total_eval_loss += batch_loss
        eval_accumulator.update(
            batch_logits=batch_logits,
            batch_loss=batch_loss,
            batch=batch,
            batch_metadata=batch_metadata,
        )

        nb_eval_examples += len(batch)
        nb_eval_steps += 1
    eval_loss = total_eval_loss / nb_eval_steps
    tokenizer = (
        jiant_model.tokenizer
        if not torch_utils.is_data_parallel(jiant_model)
        else jiant_model.module.tokenizer
    )
    output = {
        "accumulator": eval_accumulator,
        "loss": eval_loss,
        "metrics": evaluation_scheme.compute_metrics_from_accumulator(
            task=task, accumulator=eval_accumulator, labels=val_labels, tokenizer=tokenizer,
        ),
    }
    if return_preds:
        output["preds"] = evaluation_scheme.get_preds_from_accumulator(
            task=task, accumulator=eval_accumulator,
        )
    return output


def run_test(test_dataloader, jiant_model: JiantModel, task, device, local_rank, verbose=True):
    if not local_rank == -1:
        return
    jiant_model.eval()
    evaluation_scheme = evaluate.get_evaluation_scheme_for_task(task=task)
    eval_accumulator = evaluation_scheme.get_accumulator()

    for step, (batch, batch_metadata) in enumerate(
        maybe_tqdm(test_dataloader, desc=f"Eval ({task.name}, Test)", verbose=verbose)
    ):
        batch = batch.to(device)

        with torch.no_grad():
            model_output = wrap_jiant_forward(
                jiant_model=jiant_model, batch=batch, task=task, compute_loss=False,
            )
        batch_logits = model_output.logits.detach().cpu().numpy()
        eval_accumulator.update(
            batch_logits=batch_logits, batch_loss=0, batch=batch, batch_metadata=batch_metadata,
        )
    return {
        "preds": evaluation_scheme.get_preds_from_accumulator(
            task=task, accumulator=eval_accumulator,
        ),
        "accumulator": eval_accumulator,
    }
