import argparse
import os.path

import numpy as np
import pandas as pd
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
import models


def predict_joint(model, args, logger=None):
    file_name = os.path.splitext(os.path.basename(args.joint_file))[0]
    logger.log(logging.INFO, f'The {file_name} file start predicting >>>')

    joint_df = pd.read_parquet(args.joint_file)
    dataset = utils.EccDNADataset_joint_(joint_df, args.joint_length, args.genome_file, args.chr)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, persistent_workers=True,
                            pin_memory=True, drop_last=False)

    model.load_state_dict(torch.load(args.parameter_path))
    model.to(args.device)
    model.eval()

    saved_probabilities = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(torch.float32).to(args.device)
            outputs = model(inputs)
            prob = torch.sigmoid(outputs).squeeze().detach().cpu().numpy()
            prob = np.round(prob * 100).astype(np.int8)
            saved_probabilities.extend(prob)

    saved_probabilities = np.array(saved_probabilities)
    save_path_ = os.path.join(args.save_path, 'records', args.dir_name)
    os.makedirs(save_path_, exist_ok=True)
    save_name = os.path.join(save_path_, file_name + '.npy')
    np.save(save_name, saved_probabilities)
    logger.log(logging.INFO, f'The {file_name} file finish predicting >>>')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--joint_file', type=str)
    parser.add_argument('--genome_file', type=str)
    parser.add_argument('--parameter_path', type=str)
    parser.add_argument('--chr', type=str, default='chr1')
    parser.add_argument('--joint_length', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=5120)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--dir_name', type=str, default='test')
    parser.add_argument('--parameter_choice', type=str, required=True)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    now = datetime.now()
    formatted_time_ = now.strftime("%m-%d-%H-%M-%S")
    args.dir_time = formatted_time_
    logger = utils.SafeLogger(args)
    logger.log(logging.INFO, f"Start Time & Fold Name: {formatted_time_}")
    utils.record_multi(logger, utils.record_parameter(args))
    args.chr = args.chr[3:]

    logger.log(logging.INFO, 'The Predicting is Starting')
    net = models.BreakModel(seq_len=args.joint_length, num_classes=1, parameter=args.parameter_choice,
                            dropout_rate=args.dropout)

    predict_joint(net, args, logger)
    logger.log(logging.INFO, 'The Predicting is Finished !!!')