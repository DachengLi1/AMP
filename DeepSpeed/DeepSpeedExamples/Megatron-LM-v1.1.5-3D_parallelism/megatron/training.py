# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain utilities."""

from datetime import datetime
import math
import sys
import os
import time
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from apex.optimizers import FusedAdam as Adam
import numpy as np

from megatron import get_args
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import mpu
from megatron import print_rank_0
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.fp16 import FP16_Module
from megatron.fp16 import FP16_Optimizer
from megatron.initialize import initialize_megatron
from megatron.learning_rates import AnnealingLR, LinearLrDecay
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import get_params_for_weight_decay_optimization
from megatron.model.realm_model import ICTBertModel
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import make_data_loader, make_data_loader_transgan
from megatron.utils import report_memory

import deepspeed


def pretrain(train_valid_test_dataset_provider, model_provider,
             forward_step_func, extra_args_provider=None, args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    args = get_args()
    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    timers('model and optimizer').stop()

    # Data stuff.
    timers('train/valid/test data iterators').start()
    train_data_iterator, valid_data_iterator, test_data_iterator \
        = build_train_valid_test_data_iterators(
            train_valid_test_dataset_provider)
    timers('train/valid/test data iterators').stop()

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['model and optimizer', 'train/valid/test data iterators'])
    print_rank_0('training ...')

    iteration = 0
    if args.do_train and args.train_iters > 0:
        iteration = train(forward_step_func,
                          model, optimizer, lr_scheduler,
                          train_data_iterator, valid_data_iterator)

    #if args.do_valid:
    #    prefix = 'the end of training for val data'
    #    evaluate_and_print_results(prefix, forward_step_func,
    #                               valid_data_iterator, model,
    #                               iteration, False)

    #if args.save and iteration != 0:
    #    save_checkpoint(iteration, model, optimizer, lr_scheduler)

    #if args.do_test:
        # Run on test data.
    #    prefix = 'the end of training for test data'
    #    evaluate_and_print_results(prefix, forward_step_func,
    #                               test_data_iterator, model,
    #                              0, True)


def pretrain_transgan(train_valid_test_dataset_provider, model_provider,
             forward_step_func, extra_args_provider=None, args_defaults={}):
    """Main training program.
    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.
    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    args = get_args()
    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer_transgan(model_provider)
    timers('model and optimizer').stop()

    # Data stuff.
    timers('train/valid/test data iterators').start()
    train_data_iterator, valid_data_iterator, test_data_iterator \
        = build_train_valid_test_data_iterators_transgan(
            train_valid_test_dataset_provider)
    #print(f"here is {train_data_iterator} --------------------------------------- ")
    timers('train/valid/test data iterators').stop()

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['model and optimizer', 'train/valid/test data iterators'])
    print_rank_0('training ...')

    iteration = 0
    if args.do_train and args.train_iters > 0:
        iteration = train_transgan(forward_step_func,
                          model, optimizer, lr_scheduler,
                          train_data_iterator, valid_data_iterator)

    #if args.do_valid:
    #    prefix = 'the end of training for val data'
    #    evaluate_and_print_results(prefix, forward_step_func,
    #                               valid_data_iterator, model,
    #                               iteration, False)

    #if args.save and iteration != 0:
    #    save_checkpoint(iteration, model, optimizer, lr_scheduler)

    #if args.do_test:
    #    # Run on test data.
    #    prefix = 'the end of training for test data'
    #    evaluate_and_print_results(prefix, forward_step_func,
    #                               test_data_iterator, model,
    #                               0, True)

def get_model(model_provider_func):
    """Build the model."""
    args = get_args()

    # Build model on cpu.
    model = model_provider_func()

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    if args.deepspeed:
        # DeepSpeed handles CUDA, FP16, and DDP components.
        return model

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training."""
    if args.DDP_impl == 'torch':
        i = torch.cuda.current_device()
        model = torchDDP(model, device_ids=[i], output_device=i,
                         process_group=mpu.get_data_parallel_group())
        return model
    if args.DDP_impl == 'local':
        model = LocalDDP(model)
        return model

    raise NotImplementedError('Unknown DDP implementation specified: {}. '
                              'Exiting.'.format(args.DDP_impl))

def get_model_transgan(model_provider_func):
    """Build the model."""
    args = get_args()

    # Build model on cpu.
    gen_model, dis_model = model_provider_func()

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in gen_model.parameters()])), flush=True)

    if args.deepspeed:
        # DeepSpeed handles CUDA, FP16, and DDP components.
        return (gen_model, dis_model)

    # GPU allocation.
    gen_model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        gen_model = FP16_Module(gen_model)

    # Wrap model for distributed training."""
    if args.DDP_impl == 'torch':
        i = torch.cuda.current_device()
        gen_model = torchDDP(gen_model, device_ids=[i], output_device=i,
                         process_group=mpu.get_data_parallel_group())
        return (gen_model, dis_model)
    if args.DDP_impl == 'local':
        gen_model = LocalDDP(gen_model)
        return (gen_model, dis_model)

    raise NotImplementedError('Unknown DDP implementation specified: {}. '
                              'Exiting.'.format(args.DDP_impl))


def get_optimizer(model):
    """Set up the optimizer."""
    args = get_args()

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (torchDDP, LocalDDP, FP16_Module)):
        model = model.module
    param_groups = get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    if args.cpu_optimizer:
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                                       lr=args.lr,
                                       weight_decay=args.weight_decay)
    else:
        # Use Adam.
        optimizer = Adam(param_groups,
                         lr=args.lr,
                         weight_decay=args.weight_decay,
                         betas=(args.adam_beta1, args.adam_beta2),
                         eps=args.adam_eps)

    if args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer

def get_optimizer_transgan(model):
    """Set up the optimizer."""
    args = get_args()

    gen_model, dis_model = model

    # Build parameter groups (weight decay and non-decay).
    while isinstance(gen_model, (torchDDP, LocalDDP, FP16_Module)):
        gen_model = gen_model.module
    param_groups = get_params_for_weight_decay_optimization(gen_model)

    # hwang: we fix the optimizer as Adam now
    gen_optimizer = torch.optim.Adam(param_groups, args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_model.parameters()), args.d_lr, (args.beta1, args.beta2))
    # Add model parallel attribute if it is not set.
    # for param_group in param_groups:
    #     for param in param_group['params']:
    #         if not hasattr(param, 'model_parallel'):
    #             param.model_parallel = False

    # if args.cpu_optimizer:
    #     if args.cpu_torch_adam:
    #         cpu_adam_optimizer = torch.optim.Adam
    #     else:
    #         from deepspeed.ops.adam import DeepSpeedCPUAdam
    #         cpu_adam_optimizer = DeepSpeedCPUAdam
    #     optimizer = cpu_adam_optimizer(param_groups,
    #                                    lr=args.lr,
    #                                    weight_decay=args.weight_decay)
    # else:
    #     # Use Adam.
    #     optimizer = Adam(param_groups,
    #                      lr=args.lr,
    #                      weight_decay=args.weight_decay,
    #                      betas=(args.adam_beta1, args.adam_beta2),
    #                      eps=args.adam_eps)

    if args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return (gen_optimizer, dis_optimizer)

    # Wrap into fp16 optimizer.
    if args.fp16:
        gen_optimizer = FP16_Optimizer(gen_optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

        dis_optimizer = FP16_Optimizer(dis_optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return (gen_optimizer, dis_optimizer)


def get_learning_rate_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = 0
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=args.lr,
        warmup_iter=warmup_iter,
        total_iters=num_iters,
        decay_style=args.lr_decay_style,
        last_iter=init_step,
        min_lr=args.min_lr,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler)

    return lr_scheduler


def get_learning_rate_scheduler_transgan(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    gen_optimizer, dis_optimizer = optimizer
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_critic)

    # Add linear learning rate scheduler.
    # if args.lr_decay_iters is not None:
    #     num_iters = args.lr_decay_iters
    # else:
    #     num_iters = args.train_iters
    # num_iters = max(1, num_iters)
    # init_step = 0
    # warmup_iter = args.warmup * num_iters
    # lr_scheduler = AnnealingLR(
    #     optimizer,
    #     start_lr=args.lr,
    #     warmup_iter=warmup_iter,
    #     total_iters=num_iters,
    #     decay_style=args.lr_decay_style,
    #     last_iter=init_step,
    #     min_lr=args.min_lr,
    #     use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
    #     override_lr_scheduler=args.override_lr_scheduler)

    return (gen_scheduler, dis_scheduler)


def setup_model_and_optimizer(model_provider_func):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func)
    optimizer = get_optimizer(model)
    lr_scheduler = get_learning_rate_scheduler(optimizer)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu if args.pipe_parallel_size == 0 else None,
            dist_init_required=False)

        if args.pipe_parallel_size > 0:
            model.set_batch_fn(model.module._megatron_batch_fn)

    if args.load is not None:
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler)
    else:
        args.iteration = 0

    # get model without FP16 and/or TorchDDP wrappers
    unwrapped_model = model
    while hasattr(unwrapped_model, 'module'):
        unwrapped_model = unwrapped_model.module

    if args.iteration == 0 and hasattr(unwrapped_model, 'init_state_dict_from_bert'):
        print("Initializing ICT from pretrained BERT model", flush=True)
        unwrapped_model.init_state_dict_from_bert()

    return model, optimizer, lr_scheduler

def setup_model_and_optimizer_transgan(model_provider_func):
    """Setup model and optimizer."""
    args = get_args()

    gen_model, dis_model = get_model_transgan(model_provider_func)
    model = (gen_model, dis_model)
    optimizer = get_optimizer_transgan(model)
    lr_scheduler = get_learning_rate_scheduler_transgan(optimizer)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        gen_model, gen_optimizer, _, gen_lr_scheduler = deepspeed.initialize(
            model=gen_model,
            optimizer=optimizer[0], # gen_optimizer
            args=args,
            lr_scheduler=lr_scheduler[0], # lr_scheduler
            mpu=mpu if args.pipe_parallel_size == 0 else None,
            dist_init_required=False)

        if args.pipe_parallel_size > 0:
            gen_model.set_batch_fn(gen_model.module._megatron_batch_fn)

    # we comment this part out for now, will see how to support it in the future
    #if args.load is not None:
    #    args.iteration = load_checkpoint(model, optimizer, lr_scheduler)
    #else:
    #    args.iteration = 0
    args.iteration = 0

    # get model without FP16 and/or TorchDDP wrappers
    
    unwrapped_model = gen_model
    while hasattr(unwrapped_model, 'module'):
        unwrapped_model = unwrapped_model.module

    #if args.iteration == 0 and hasattr(unwrapped_model, 'init_state_dict_from_bert'):
    #    print("Initializing ICT from pretrained BERT model", flush=True)
    #    unwrapped_model.init_state_dict_from_bert()

    return (gen_model, dis_model), optimizer, lr_scheduler

def backward_step(optimizer, model, loss):
    """Backward step."""
    args = get_args()
    timers = get_timers()

    # Backward pass.
    #timers('backward-backward').start()
    if args.deepspeed:
        model.backward(loss)
    else:
        optimizer.zero_grad(set_grads_to_None=True)
        if args.fp16:
            optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()
    #timers('backward-backward').stop()

    if args.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers('backward-allreduce').reset()
    else:
        # All-reduce if needed.
        if args.DDP_impl == 'local':
            # Always use deepspeed 
            assert False
            timers('backward-allreduce').start()
            model.allreduce_params(reduce_after=False,
                                   fp32_allreduce=args.fp32_allreduce)
            timers('backward-allreduce').stop()

    if not args.deepspeed:
        # Update master gradients.
        timers('backward-master-grad').start()
        if args.fp16:
            optimizer.update_master_grads()
        timers('backward-master-grad').stop()

        # Clipping gradients helps prevent the exploding gradient.
        timers('backward-clip-grad').start()
        if args.clip_grad > 0:
            if not args.fp16:
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)
        timers('backward-clip-grad').stop()

def train_step_transgan(forward_step_func, data_iterator,
               model, optimizer, lr_scheduler):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Pipeline parallelism schedules forward/backward/step
    if args.pipe_parallel_size > 0:
        return train_step_pipe(model, data_iterator)

    # Forward model for one step.
    timers('forward').start()
    loss, loss_reduced = forward_step_func(data_iterator, model)
    timers('forward').stop()
    # Calculate gradients, reduce across processes, and clip.
    timers('backward').start()
    backward_step(optimizer, model, loss)
    timers('backward').stop()

    # Update parameters.
    skipped_iter = 0
    timers('optimizer').start()
    if args.deepspeed:
        model.step()
    else:
        optimizer.step()
        # Update learning rate.
        if not (args.fp16 and optimizer.overflow):
            lr_scheduler.step()
        else:
            skipped_iter = 1
    timers('optimizer').stop()

    return loss_reduced, skipped_iter


def train_step(forward_step_func, data_iterator,
               model, optimizer, lr_scheduler):
    """Single training step."""
    #assert False
    args = get_args()
    timers = get_timers()

    # Pipeline parallelism schedules forward/backward/step
    if args.pipe_parallel_size > 0:
        return train_step_pipe(model, data_iterator)

    # Forward model for one step.
    timers('forward').start()
    loss, loss_reduced = forward_step_func(data_iterator, model)
    timers('forward').stop()

    # Calculate gradients, reduce across processes, and clip.
    timers('backward').start()
    backward_step(optimizer, model, loss)
    timers('backward').stop()

    # Update parameters.
    skipped_iter = 0
    timers('optimizer').start()
    if args.deepspeed:
        model.step()
    else:
        optimizer.step()
        # Update learning rate.
        if not (args.fp16 and optimizer.overflow):
            lr_scheduler.step()
        else:
            skipped_iter = 1
    timers('optimizer').stop()

    return loss_reduced, skipped_iter

def train_step_pipe(model, data_iterator):
    """Single training step with DeepSpeed's pipeline parallel engine. """
    args = get_args()
    timers = get_timers()

    assert args.deepspeed
    loss = model.train_batch(data_iter=data_iterator)
    loss_dict = {'lm loss': loss}
    if args.fp16 and model.optimizer.overflow:
        skipped_iter = 1
    else:
        skipped_iter = 0

    # Don't break Megatron's timers because we changed code paths.
    for t in ['forward', 'backward', 'allreduce', 'optimizer', 'batch generator',
              'data loader']:
        timers(t).reset()
    return loss_dict, skipped_iter



def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Update losses.
    skipped_iters_key = 'skipped iterations'
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    got_nan_key = 'got nan'

    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(key, 0.) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan

    total_loss_dict[got_nan_key] = total_loss_dict.get(
        got_nan_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = []

    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)
    #add_to_logging('forward')
    #add_to_logging('backward')
    #add_to_logging('backward-backward')
    #add_to_logging('backward-allreduce')
    #add_to_logging('backward-master-grad')
    #add_to_logging('backward-clip-grad')
    #add_to_logging('optimizer')
    add_to_logging('batch generator')
    add_to_logging('mp allreduce')

    # Tensorboard values.
    if writer and torch.distributed.get_rank() == 0:
        writer.add_scalar('learning_rate', learning_rate, iteration)
        for key in loss_dict:
            writer.add_scalar(key, loss_dict[key], iteration)
        #if args.fp16:
        #    writer.add_scalar('loss_scale', loss_scale, iteration)
        normalizer = iteration % args.log_interval
        if normalizer == 0:
            normalizer = args.log_interval
        timers.write(timers_to_log, writer, iteration,
                     normalizer=normalizer)
    if iteration % args.log_interval == 0:
        elapsed_time, elapsed_std = timers('interval time').elapsed()
        if writer and torch.distributed.get_rank() == 0:
            writer.add_scalar('iteration_time',
                              elapsed_time / args.log_interval, iteration)
        log_string = ' iteration {:8d}/{:8d} |'.format(iteration,
                                                       args.train_iters)
        log_string += ' elapsed time per iteration (ms): {:.1f} ({:.1f})|'.format(
            elapsed_time * 1000.0 / args.log_interval, elapsed_std * 1000.0)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        num_iterations = max(
            1, args.log_interval - total_loss_dict[skipped_iters_key])
        for key in total_loss_dict:
            if key not in [skipped_iters_key, got_nan_key]:
                avg = total_loss_dict[key] / float(num_iterations)
                log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = 0.0
        #if args.fp16:
        #    log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[got_nan_key])
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[got_nan_key] = 0
        print_rank_0(log_string)
        if report_memory_flag:
            report_memory('after {} iterations'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def train(forward_step_func, model, optimizer, lr_scheduler,
          train_data_iterator, valid_data_iterator):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    report_memory_flag = True
    
    time_amp = 0
    os.environ["mp_count_param"] = str(0)
    while iteration < 41:# args.train_iters:
        #print(iteration)
        os.environ["amp_iter"] = str(iteration)
        if iteration == 20:
            time_amp = time.time()

        if iteration == 40 and torch.distributed.get_rank() == 0:
            time_used = round((time.time() - time_amp) / 20,2)
            # record time used here
            
            exp_name = args.exp_name
            home_path = os.environ['HOME']
            dir_path = os.path.join(home_path, "amp_simulate")
            #assert os.path.isdir(dir_path)
            #    os.mkdir(dir_path)
            record_path = os.path.join(dir_path, f"{exp_name}.txt")
            #print(f"checking record_path exist: {record_path}")
            #assert os.path.exists(record_path)
            #    with open(record_path, "w") as tf:
            #        pass
            if os.path.exists(record_path):
                f = open(record_path, "a")
                f.write(str(time_used) + "\n")
                f.close()

        timers('interval time').start()
        loss_dict, skipped_iter = train_step(forward_step_func,
                                             train_data_iterator,
                                             model,
                                             optimizer,
                                             lr_scheduler)
        param_ = os.environ["mp_count_param"]
        print(f"single forward has mp param {param_}")
        os.environ["mp_count_param"] = str(0)
        iteration += 1

        timers('interval time').stop()
        
        # Logging.
        loss_scale = None
        if args.fp16:
            loss_scale = optimizer.cur_scale if args.deepspeed else optimizer.loss_scale
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter)

        # Autoresume
        if args.adlr_autoresume and \
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              lr_scheduler)

        # Checkpointing
        if args.save and args.save_interval and \
           iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, lr_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
           args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(prefix, forward_step_func,
                                       valid_data_iterator, model,
                                       iteration, False)

        if args.exit_interval and iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print_rank_0('rank: {} | time: {} | exiting the program at '
                         'iteration {}'.format(rank, time_str, iteration))
            sys.exit()

    return iteration


def train_transgan(forward_step_func, model, optimizer, lr_scheduler,
          train_data_iterator, valid_data_iterator):
    """Train the model function."""
    args = get_args()
    timers = get_timers()
    
    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)
    add_to_logging('mp allreduce')

    gen_model, dis_model = model
    gen_optimizer, dis_optimizer = optimizer
    gen_lr_scheduler, dis_lr_scheduler = lr_scheduler

    # Turn on training mode which enables dropout.
    gen_model.train()
    dis_model.train()


    train_data_iterator_gen, train_data_iterator_dis = train_data_iterator

    # we will first need to run one iteration of discriminator
    dis_optimizer.zero_grad()
    #imgs, _ = next(train_data_iterator)

    #real_imgs = imgs.type(torch.cuda.FloatTensor).cuda(args.gpu, non_blocking=True)
    #z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))).cuda(args.gpu, non_blocking=True)

    # ---------------------
    #  Train Discriminator
    # ---------------------

    #real_validity = dis_model(real_imgs)
    #fake_imgs = gen_model(z, epoch).detach()
    #assert fake_imgs.size() == real_imgs.size(), f"fake_imgs.size(): {fake_imgs.size()} real_imgs.size(): {real_imgs.size()}"

    #fake_validity = dis_net(fake_imgs)

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    report_memory_flag = True
    while iteration < 41:# args.train_iters:
        os.environ["amp_iter"] = str(iteration)
        #print(iteration)
        if iteration == 20:
            time_amp = time.time()

        if iteration == 40 and torch.distributed.get_rank() == 0:
            time_used = round((time.time() - time_amp) / 20,2)
            # record time used here

            exp_name = args.exp_name
            home_path = os.environ['HOME']
            dir_path = os.path.join(home_path, "amp_simulate")
            #assert os.path.isdir(dir_path)
            #    os.mkdir(dir_path)
            record_path = os.path.join(dir_path, f"{exp_name}.txt")
            #assert os.path,exists(record_path)
            #    with open(record_path, "w") as tf:
            #        pass
            if os.path.exists(record_path):
                f = open(record_path, "a")
                f.write(str(time_used) + "\n")
                f.close()
        timers('interval time').start()

        loss_dict, skipped_iter = train_step_transgan(forward_step_func,
                                             train_data_iterator_gen,
                                             gen_model,
                                             gen_optimizer,
                                             gen_lr_scheduler)
        #assert False
        timers('interval time').stop()
        
        iteration += 1

        # Logging.
        # hongyi: let's comment this part out for now
        loss_scale = None
        #if args.fp16:
        #    #loss_scale = optimizer.cur_scale if args.deepspeed else optimizer.loss_scale
        #    loss_scale = gen_optimizer.cur_scale if args.deepspeed else gen_optimizer.loss_scale
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          gen_optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter)

        # Autoresume
        #if args.adlr_autoresume and \
        #   (iteration % args.adlr_autoresume_interval == 0):
        #    check_adlr_autoresume_termination(iteration, gen_model, gen_optimizer,
        #                                      lr_scheduler)

        # Checkpointing
        #if args.save and args.save_interval and \
        #   iteration % args.save_interval == 0:
        #    save_checkpoint(iteration, gen_model, gen_optimizer, lr_scheduler)

        # Evaluation
        #if args.eval_interval and iteration % args.eval_interval == 0 and \
        #   args.do_valid:
        #    prefix = 'iteration {}'.format(iteration)
        #    evaluate_and_print_results(prefix, forward_step_func,
        #                               valid_data_iterator, model,
        #                               iteration, False)

        if args.exit_interval and iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print_rank_0('rank: {} | time: {} | exiting the program at '
                         'iteration {}'.format(rank, time_str, iteration))
            sys.exit()

    return iteration


def evaluate(forward_step_func, data_iterator, model, verbose=False):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))
            # Forward evaluation.
            _, loss_dict = forward_step_func(data_iterator, model)

            # When contiguous memory optimizations are enabled, the buffers
            # allocated by the optimizations are deallocated during backward pass
            # in the absence of backward pass the buffers should be reset after each
            # forward pass
            if args.deepspeed and args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

            # Reduce across processes.
            for key in loss_dict:
                total_loss_dict[key] = total_loss_dict.get(key, 0.) + \
                    loss_dict[key]
    # Move model back to the train mode.
    model.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters

    return total_loss_dict


def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    writer = get_tensorboard_writer()

    # Pipeline parallelism needs eval_batch() instead of a simple forward().
    args = get_args()
    if args.pipe_parallel_size > 0:
        def _eval_helper(data_iter, pipe_model):
            loss = model.eval_batch(data_iter)
            return None, {'lm loss' : loss}
        forward_step_func = _eval_helper

    total_loss_dict = evaluate(forward_step_func, data_iterator, model, verbose)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer and torch.distributed.get_rank() == 0:
            writer.add_scalar('{} value'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} ppl'.format(key), ppl, iteration)

    length = len(string) + 1
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)
def build_train_valid_test_data_iterators_transgan(
        build_train_valid_test_datasets_provider):
    """XXX"""
    args = get_args()

    (train_dataloader_gen, train_dataloader_dis, valid_dataloader, test_dataloader) = (None, None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Ensure only the first/last pipeline stages have data loaders
    if args.pipe_parallel_size > 0:
        is_first_stage = mpu.get_pipe_parallel_rank() == 0
        is_last_stage = mpu.get_pipe_parallel_rank() == mpu.get_pipe_parallel_world_size() - 1
        pipe_load = is_first_stage or is_last_stage
    else:
        pipe_load = True

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0 and pipe_load:
        # Rank, size, and global batch size.
        data_parallel_size = mpu.get_data_parallel_world_size()
        global_batch_size = args.batch_size * data_parallel_size * args.gas

        # Number of train/valid/test samples.
        train_iters = args.train_iters
        eval_iters = (train_iters // args.eval_interval + 1) * args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_iters * global_batch_size,
                                      eval_iters * global_batch_size,
                                      test_iters * global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.
        (train_ds_gen, train_ds_dis), valid_ds, test_ds = build_train_valid_test_datasets_provider(
            args=args)

        # def make_data_loader(dataset):
        #     """Buld dataloader given an input dataset."""
        #     if dataset is None:
        #         return None
        #     args = get_args()

        #     # Data parallel arguments.
        #     world_size = mpu.get_data_parallel_world_size()
        #     rank = mpu.get_data_parallel_rank()
        #     global_batch_size = args.batch_size * world_size
        #     num_workers = args.num_workers

        #     # Use a simple sampler with distributed batch sampler.
        #     sampler = torch.utils.data.SequentialSampler(dataset)
        #     batch_sampler = DistributedBatchSampler(sampler=sampler,
        #                                             batch_size=global_batch_size,
        #                                             drop_last=True,
        #                                             rank=rank,
        #                                             world_size=world_size)
        #     # Torch dataloader.
        #     return torch.utils.data.DataLoader(dataset,
        #                                        batch_sampler=batch_sampler,
        #                                        num_workers=num_workers,
        #                                        pin_memory=True)

        # Build dataloders.
        train_dataloader_gen = make_data_loader_transgan(train_ds_gen)
        #train_dataloader_dis = make_data_loader_transgan(train_ds_dis.train_dataset)
        #valid_dataloader = make_data_loader(valid_ds)
        #test_dataloader = make_data_loader(test_ds)

        valid_dataloader = None
        test_dataloader = None

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader_gen is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    if args.pipe_parallel_size > 0:
        # Only first/last pipeline stages have data loaders, so pipeline parallelism should
        # broadcast globally instead of just the model parallel group.
        torch.distributed.broadcast(flags, src=0)
    else:
        torch.distributed.broadcast(flags,
                                    mpu.get_model_parallel_src_rank(),
                                    group=mpu.get_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    # Shift the start iterations.
    if train_dataloader_gen is not None:
        train_dataloader_gen.batch_sampler.start_iter = args.iteration % \
            len(train_dataloader_gen)
        #train_dataloader_dis.batch_sampler.start_iter = args.iteration % \
        #    len(train_dataloader_dis)
        print_rank_0('setting training data start iteration to {}'.
                     format(train_dataloader_gen.batch_sampler.start_iter))
    if valid_dataloader is not None:
        start_iter_val = (args.iteration // args.eval_interval) * \
            args.eval_iters
        valid_dataloader.batch_sampler.start_iter = start_iter_val % \
            len(valid_dataloader)
        print_rank_0('setting validation data start iteration to {}'.
                     format(valid_dataloader.batch_sampler.start_iter))

    # Build iterators.
    if train_dataloader_gen is not None:
        train_data_iterator_gen = iter(train_dataloader_gen)
    else:
        train_data_iterator_gen = None
        
    #if train_dataloader_dis is not None:
    #    train_data_iterator_dis = iter(train_dataloader_dis)
    #else:
    #    train_data_iterator_dis = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader)
    else:
        test_data_iterator = None

    return (train_data_iterator_gen, None), valid_data_iterator, test_data_iterator

def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """XXX"""
    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Ensure only the first/last pipeline stages have data loaders
    if args.pipe_parallel_size > 0:
        is_first_stage = mpu.get_pipe_parallel_rank() == 0
        is_last_stage = mpu.get_pipe_parallel_rank() == mpu.get_pipe_parallel_world_size() - 1
        pipe_load = is_first_stage or is_last_stage
    else:
        pipe_load = True

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0 and pipe_load:
        # Rank, size, and global batch size.
        data_parallel_size = mpu.get_data_parallel_world_size()
        global_batch_size = args.batch_size * data_parallel_size * args.gas

        # Number of train/valid/test samples.
        train_iters = args.train_iters
        eval_iters = (train_iters // args.eval_interval + 1) * args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_iters * global_batch_size,
                                      eval_iters * global_batch_size,
                                      test_iters * global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
            train_val_test_num_samples)

        # Build dataloders.
        train_dataloader = make_data_loader(train_ds)
        valid_dataloader = make_data_loader(valid_ds)
        test_dataloader = make_data_loader(test_ds)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    if args.pipe_parallel_size > 0:
        # Only first/last pipeline stages have data loaders, so pipeline parallelism should
        # broadcast globally instead of just the model parallel group.
        torch.distributed.broadcast(flags, src=0)
    else:
        torch.distributed.broadcast(flags,
                                    mpu.get_model_parallel_src_rank(),
                                    group=mpu.get_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    # Shift the start iterations.
    if train_dataloader is not None:
        train_dataloader.batch_sampler.start_iter = args.iteration % \
            len(train_dataloader)
        print_rank_0('setting training data start iteration to {}'.
                     format(train_dataloader.batch_sampler.start_iter))
    if valid_dataloader is not None:
        start_iter_val = (args.iteration // args.eval_interval) * \
            args.eval_iters
        valid_dataloader.batch_sampler.start_iter = start_iter_val % \
            len(valid_dataloader)
        print_rank_0('setting validation data start iteration to {}'.
                     format(valid_dataloader.batch_sampler.start_iter))

    # Build iterators.
    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator
