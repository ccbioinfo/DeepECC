import os
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
import h5py
import argparse
from datetime import datetime

import models
import utils


def scan_genome_single_chr(model, args, logger=None):

    logger.log(logging.INFO, f'The Scan Starting !!!')
    model.load_state_dict(torch.load(args.parameter_path))
    model.to(args.device)
    model.eval()
    threshold_tensor = torch.tensor(args.threshold, dtype=torch.float32, device=args.device)
    result_save_path = os.path.join(args.save_path, 'records', args.dir_name, args.dir_time)
    os.makedirs(result_save_path, exist_ok=True)
    result_file = os.path.join(result_save_path, f'scan_{args.chr}_{args.start_end}.parquet')

    saved_probabilities = []
    true_counts = 0
    genome_dataset = utils.GenomeDataset(genome_file=args.genome_file, chr=args.chr, length=args.seq_len)
    genome_dataloader = DataLoader(genome_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                       pin_memory=True, num_workers=args.num_workers, persistent_workers=(args.device != 'cpu'))
    num_batches = len(genome_dataloader)
    logger.log(logging.INFO, f'The {args.chr} have {num_batches} batches')
    with torch.no_grad():
        for batch_idx, input in enumerate(genome_dataloader):
            input = input.to(torch.float32).to(args.device)
            output = model(input)

            probs = torch.sigmoid(output)
            true_counts += (probs > threshold_tensor).sum().item()
            saved_probabilities.append(probs.detach().cpu().numpy())
            logger.log(logging.INFO, f'The {batch_idx} fulfilled')
    saved_probabilities = np.concatenate(saved_probabilities)
    saved_probabilities = pd.DataFrame(saved_probabilities, columns=['probs']).astype('float32')
    saved_probabilities.to_parquet(result_file, engine='pyarrow')

    total_count = utils.chromosome_length[args.chr]
    true_rate = round(true_counts / total_count, 4)

    logger.log(logging.INFO, f'The {args.chr} finished ------------')
    logger.log(logging.INFO, f'Have {true_counts} reached threshold')
    logger.log(logging.INFO, f'The Chromosome rate is {true_rate}')
    logger.log(logging.INFO, f"Results saved to {result_save_path}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--genome_file', type=str, default='./')
    args.add_argument('--parameter_path', type=str, default='./')
    args.add_argument('--parameter_choice', type=str, default='RU_2')
    args.add_argument('--start_end', type=str, default='start')
    args.add_argument('--dropout', type=float, default=0.1)
    args.add_argument('--seq_len', type=int, default=1500)
    args.add_argument('--batch_size', type=int, default=1024)
    args.add_argument('--num_workers', type=int, default=8)
    args.add_argument('--threshold', type=float, default=0.8)
    args.add_argument('--chr', type=str, default='chr1')
    args.add_argument('--save_path', type=str, default='./')
    args.add_argument('--dir_name', type=str, default='test')
    args.add_argument('--device', type=str, default='cpu')

    args = args.parse_args()
    now = datetime.now()
    formatted_time_ = now.strftime("%m-%d-%H-%M-%S")
    args.dir_time = formatted_time_
    logger = utils.SafeLogger(args)
    logger.log(logging.INFO, f"Start Time & Fold Name: {formatted_time_}")
    utils.record_multi(logger, utils.record_parameter(args))


    net = models.BreakModel(seq_len=args.seq_len, num_classes=1, parameter=args.parameter_choice,
                            dropout_rate=args.dropout)
    scan_genome_single_chr(net, args, logger)
    logger.log(logging.INFO, f"Finish Scan !!!")








