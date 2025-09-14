import gc
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
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from sklearn.model_selection import KFold

import utils
import train_parallel

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def get_dataloader_fold(args):
    data = utils.load_data(args.pos_data, args.neg_data, shuffled=True)
    dataset = utils.EccDNADataset(data, start_end=args.start_end)

    kflod = KFold(n_splits=5, shuffle=True, random_state=seed)
    indices = np.arange(len(dataset))
    all_loaders = []

    for i, (train_idx, valid_idx) in enumerate(kflod.split(indices)):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        valid_subset = torch.utils.data.Subset(dataset, valid_idx)


        train_sampler = DistributedSampler(train_subset, shuffle=True)
        valid_sampler = DistributedSampler(valid_subset, shuffle=False)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                  sampler=train_sampler, drop_last=True,
                                  pin_memory=True, num_workers=0)
        valid_loader = DataLoader(valid_subset, batch_size=args.batch_size,
                                  sampler=valid_sampler, drop_last=True,
                                  pin_memory=True, num_workers=0)
        all_loaders.append((train_loader, valid_loader))

    return all_loaders


def main_worker_fold_valid(rank, world_size, args):
    train_parallel.setup(rank, world_size)
    logger = utils.SafeLogger(args)
    if rank == 0:
        logger.log(logging.INFO, 'The Training is Starting')
    for fold, (train_loader, valid_loader) in enumerate(get_dataloader_fold(args)):

        model = train_parallel.get_model(args, rank)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        if args.pos_weight is not None:
            pos_weight = torch.tensor([args.pos_weight], device=torch.cuda.current_device())
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        if rank == 0:
            logger.log(logging.INFO, '************')
            logger.log(logging.INFO, f'This is {fold} Fold Validation !!!!')
            logger.log(logging.INFO, '************')

        best_val_loss = float('inf')
        best_model_state_dict = None
        best_epoch = 0
        metrics_executor = ThreadPoolExecutor(max_workers=2)

        patience = 5
        no_improve_count = 0
        min_delta = 0.0001

        for epoch in range(args.epochs):

            early_stop = torch.tensor(0, device=torch.device('cuda', rank))
            train_loss_mean, train_all_probs, train_all_preds, train_all_targets = train_parallel.train(model, train_loader, optimizer,
                                                                                                        criterion, rank, epoch, args)
            val_loss_mean, val_all_probs, val_all_preds, val_all_targets = train_parallel.evaluate(model, valid_loader, criterion,
                                                                                                   rank, epoch, args)
            if rank == 0:
                cpu_data = train_parallel.prepare_for_thread(epoch,
                                                             train_loss_mean, val_loss_mean,
                                                             train_all_targets, train_all_preds,
                                                             val_all_targets, val_all_preds
                                                             )
                metrics_contents = utils.show_metrics(*cpu_data)
                utils.record_multi(logger, metrics_contents)

                cpu_probs = train_parallel.prepare_for_thread(epoch,
                                                              train_all_targets, train_all_probs,
                                                              val_all_targets, val_all_probs, args.threshold, args.save_path,
                                                              args.dir_name, args.dir_time
                                                              )
                cpu_probs_ = train_parallel.prepare_for_thread(fold,
                                                               train_all_targets, train_all_probs,
                                                               val_all_targets, val_all_probs, args.threshold, args.save_path,
                                                               args.dir_name, args.dir_time
                                                               )

                metrics_executor.submit(train_parallel.safe_submit, utils.save_prob_distribution, *cpu_probs)
                if epoch % 10 == 0 :
                    logger.log(logging.INFO, f"LR: {optimizer.param_groups[0]['lr']}")

                if val_loss_mean < best_val_loss - min_delta:
                    best_val_loss = val_loss_mean
                    best_model_state_dict = model.module.state_dict()
                    best_epoch = epoch
                    no_improve_count = 0
                    metrics_executor.submit(train_parallel.safe_submit, utils.save_prob_records_, *cpu_probs_)
                else:
                    no_improve_count += 1

                    if no_improve_count >= patience:
                        logger.log(logging.INFO, 'Reached patience, stop training')
                        early_stop.fill_(1)
            dist.broadcast(early_stop, src=0)
            if early_stop.item() == 1:
                break
        metrics_executor.shutdown(wait=True)
        if rank == 0 and best_model_state_dict is not None:
            utils.save_weights(logger, best_model_state_dict, save_path=args.save_path, dir_name=args.dir_name, dir_time=args.dir_time)
            utils.record_end(logger, best_epoch)
            logger.log(logging.INFO, '############')
            logger.log(logging.INFO, f'The {fold} Fold Validation is Finished !!!')
            logger.log(logging.INFO, '############')

        del model, optimizer, criterion, best_model_state_dict
        torch.cuda.empty_cache()
        gc.collect()

    dist.barrier()
    train_parallel.cleanup()


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
    args.add_argument('--smoothing', type=float, default=0.1)
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

    mp.spawn(main_worker_fold_valid,
             args=(args.world_size, args),
             nprocs=args.world_size,
             join=True)

    logger.log(logging.INFO, '→→→→→→→→→→→→→  Fold Validation is Finished !!!  ←←←←←←←←←←←←←')










