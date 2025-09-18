import os
import argparse
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

import utils
import models
from . import valid


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def get_model(args, rank):
    model = models.BreakModel(seq_len=args.seq_len, num_classes=1,
                              parameter=args.parameter_choice,
                              dropout_rate=args.dropout)
    model.cuda(rank)
    model = DDP(model, device_ids=[rank])
    return model

def get_dataloader_train(args):
    data = utils.load_data(args.pos_data, args.neg_data, shuffled=True, joint=args.joint)
    dataset = utils.EccDNADataset(data, start_end=args.start_end)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, drop_last=True,
                              pin_memory=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            sampler=val_sampler, drop_last=True,
                            pin_memory=True, num_workers=2, persistent_workers=True)


    return train_loader, val_loader


def gather_ddp_tensors(prob_list, pred_list, target_list, loss_list, rank):
    local_probs = torch.cat(prob_list, dim=0)
    local_preds = torch.cat(pred_list, dim=0)
    local_targets = torch.cat(target_list, dim=0)
    local_losses_tensor = torch.tensor(loss_list, device='cuda')

    world_size = dist.get_world_size()
    gathered_probs = [torch.zeros_like(local_probs) for _ in range(world_size)]
    gathered_preds = [torch.zeros_like(local_preds) for _ in range(world_size)]
    gathered_targets = [torch.zeros_like(local_targets) for _ in range(world_size)]
    gathered_losses = [torch.zeros_like(local_losses_tensor) for _ in range(world_size)]

    dist.all_gather(gathered_probs, local_probs)
    dist.all_gather(gathered_preds, local_preds)
    dist.all_gather(gathered_targets, local_targets)
    dist.all_gather(gathered_losses, local_losses_tensor)

    if rank == 0:
        all_probs = torch.cat(gathered_probs, dim=0)
        all_preds = torch.cat(gathered_preds, dim=0)
        all_targets = torch.cat(gathered_targets, dim=0)
        all_losses = torch.cat(gathered_losses, dim=0)
        loss_mean = all_losses.mean().item()
        return loss_mean, all_probs, all_preds, all_targets
    else:
        return None, None, None, None


def train(model, train_loader, optimizer, criterion, rank, epoch, args):

    model.train()
    local_preds = []
    local_targets = []
    local_losses = []
    local_probs = []
    threshold_tensor = torch.tensor(args.threshold).cuda()
    train_loader.sampler.set_epoch(epoch)

    for input, label in train_loader:

        input = input.to(torch.float32).cuda(rank)
        label = label.to(torch.float32).cuda(rank)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        local_losses.append(loss.item())
        prob = torch.sigmoid(output)
        # pred = (prob > args.threshold)
        pred = (prob > threshold_tensor)
        local_probs.append(prob)
        local_preds.append(pred)
        local_targets.append(label)

    return gather_ddp_tensors(local_probs, local_preds, local_targets, local_losses, rank)


def evaluate(model, val_loader, criterion, rank, epoch, args):

    model.eval()
    local_preds = []
    local_targets = []
    local_losses = []
    local_probs = []
    threshold_tensor = torch.tensor(args.threshold).cuda()

    with torch.no_grad():
        for input, label in val_loader:
            input = input.to(torch.float32).cuda(rank)
            label = label.to(torch.float32).cuda(rank)
            output = model(input)
            loss = criterion(output, label)
            local_losses.append(loss.item())
            prob = torch.sigmoid(output)
            pred = (prob > threshold_tensor)
            local_probs.append(prob)
            local_preds.append(pred)
            local_targets.append(label)

    return gather_ddp_tensors(local_probs, local_preds, local_targets, local_losses, rank)


def prepare_for_thread(*tensors):
    results = []
    for x in tensors:
        if isinstance(x, torch.Tensor):
            results.append(x.detach().cpu().numpy())
        else:
            results.append(x)
    return results


def safe_submit(func, *args):
    try:
        func(*args)
    except Exception as e:
        print(f"[Error in thread] {str(e)}")


def main_worker(rank, world_size, args):
    setup(rank, world_size)
    logger = utils.SafeLogger(args)
    if rank == 0:
        # utils.setup_logging(args.dir_name, args.dir_time)
        # utils.record_parameter(args)
        # utils.record_content('The Training is Starting')
        logger.log(logging.INFO, 'The Training is Starting')
    model = get_model(args, rank)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    if args.pos_weight is not None:
        pos_weight = torch.tensor([args.pos_weight], device=torch.cuda.current_device())
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    train_loader, val_loader = get_dataloader_train(args)

    best_val_loss = float('inf')
    best_model_state_dict = None
    best_epoch = 0
    metrics_executor = ThreadPoolExecutor(max_workers=2)

    patience = 5
    no_improve_count = 0
    min_delta = 0.0001

    for epoch in range(args.epochs):

        early_stop = torch.tensor(0, device=torch.device('cuda', rank))
        train_loss_mean, train_all_probs, train_all_preds, train_all_targets = train(model, train_loader, optimizer,
                                                                                    criterion, rank, epoch, args)
        val_loss_mean, val_all_probs, val_all_preds, val_all_targets = evaluate(model, val_loader, criterion,
                                                                                        rank, epoch, args)
        # scheduler.step()
        if rank == 0:
            cpu_data = prepare_for_thread(epoch,
                train_loss_mean, val_loss_mean,
                train_all_targets, train_all_preds,
                val_all_targets, val_all_preds
            )
            # metrics_executor.submit(safe_submit, utils.show_metrics, *cpu_data)
            # metrics_contents = utils.show_metrics(epoch, train_loss_mean, val_loss_mean, train_all_targets, train_all_preds, val_all_targets, val_all_preds)
            metrics_contents = utils.show_metrics(*cpu_data)
            utils.record_multi(logger, metrics_contents)

            cpu_probs = prepare_for_thread(epoch,
                train_all_targets, train_all_probs,
                val_all_targets, val_all_probs, args.threshold, args.save_path,
                args.dir_name, args.dir_time
            )
            metrics_executor.submit(safe_submit,utils.save_prob_distribution, *cpu_probs)
            # utils.save_prob_distribution(epoch, train_all_targets, train_all_probs, val_all_targets, val_all_probs, args.threshold)
            if epoch % 10 == 0 :
                # utils.record_content(f"LR: {optimizer.param_groups[0]['lr']}")
                logger.log(logging.INFO, f"LR: {optimizer.param_groups[0]['lr']}")

            if val_loss_mean < best_val_loss - min_delta:
                best_val_loss = val_loss_mean
                best_model_state_dict = model.module.state_dict()
                best_epoch = epoch
                no_improve_count = 0
                metrics_executor.submit(safe_submit, utils.save_prob_records, *cpu_probs)
            else:
                no_improve_count += 1

                if no_improve_count >= patience:
                    # utils.record_content('Reached patience, stop training')
                    logger.log(logging.INFO, 'Reached patience, stop training')
                    early_stop.fill_(1)
        dist.broadcast(early_stop, src=0)
        if early_stop.item() == 1:
            break
    metrics_executor.shutdown(wait=True)
    if rank == 0 and best_model_state_dict is not None:
        utils.save_weights(logger, best_model_state_dict, save_path=args.save_path, dir_name=args.dir_name, dir_time=args.dir_time)
        utils.record_end(logger, best_epoch)
    dist.barrier()
    cleanup()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--pos_data', type=str, default=' ')
    args.add_argument('--neg_data', type=str, default=' ')
    args.add_argument('--start_end', type=str, default='start')
    args.add_argument('--seq_len', type=int, default=1500)
    args.add_argument('--joint', action='store_true')
    args.add_argument('--parameter_choice', type=str, default='RU_2')
    args.add_argument('--epochs', type=int, default=200)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--wd', type=float, default=0)
    args.add_argument('--pos_weight', type=float, default=None)
    args.add_argument('--dropout', type=float, default=0.1)
    args.add_argument('--batch_size', type=int, default=10240)
    args.add_argument('--smoothing', type=float, default=0)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--val_split', type=float, default=0.2)
    args.add_argument('--show', action='store_true')
    args.add_argument('--save_path', type=str, default='./')
    args.add_argument('--dir_name', type=str, default='test')
    args.add_argument('--world_size', type=int, default=3)
    args.add_argument('--device', type=str, default='cpu')


    args = args.parse_args()
    now = datetime.now()
    formatted_time_ = now.strftime("%m-%d-%H-%M-%S")
    args.dir_time = formatted_time_

    logger = utils.SafeLogger(args)
    logger.log(logging.INFO, f"Start Time & Fold Name: {formatted_time_}")
    utils.record_multi(logger, utils.record_parameter(args))

    mp.spawn(main_worker,
             args=(args.world_size, args),
             nprocs=args.world_size,
             join=True)
