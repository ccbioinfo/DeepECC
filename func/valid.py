import logging
import os
import argparse
import psutil
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
import h5py
from datetime import datetime

import models
import utils
from . import train


def valid_model(model, dataset, parameter_path, threshold=0.5, batch_size=2048, device='cpu', args=None, logger=None, df=False, parallel=False, neg_threshold=None,):

    # model.load_state_dict(torch.load(parameter_path))

    state_dict = torch.load(parameter_path)
    if 'module.' in next(iter(state_dict)):
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    if parallel:
        device_ids = list(range(torch.cuda.device_count()))
        model = DataParallel(model, device_ids=device_ids).to(device_ids[0])
        device = device_ids[0]
    else:
        model.to(device)

    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                            num_workers=16, pin_memory=True, persistent_workers=True)

    valid_true_label = []
    valid_predict_label = []
    valid_probs = []
    with torch.no_grad():
        for input, label in dataloader:
            input = input.to(torch.float32).to(device)
            label = label.to(torch.float32).to(device)
            output = model(input)

            prob = torch.sigmoid(output)
            valid_probs.extend(prob.tolist())
            temp_predict_ = (prob > threshold)
            valid_predict_label.extend(temp_predict_.tolist())
            valid_true_label.extend(label.tolist())

    val_acc, val_recall, val_precision, val_f1, val_mcc = utils.compute_metrics(valid_true_label,
                                                                            valid_predict_label)

    metric_content = utils.show_metrics_valid(val_acc, val_recall, val_precision, val_f1, val_mcc)
    if logger is not None:
        logger.log(logging.INFO, metric_content)

    utils.save_prob_distribution_valid(valid_true_label, valid_probs, threshold,
                                 save_path=args.save_path, dir_name=args.dir_name, dir_time=args.dir_time)

    if df:
        if neg_threshold is not None:
            valid_probs = np.array(valid_probs)
            if neg_threshold == 0:
                predict_error_index = [index for index, value in enumerate(valid_probs) if 0.5 <= value <= 0.8 ]
                logger.log(logging.INFO, 'Using the None threshold ')
            else:
                predict_error_index = [index for index, value in enumerate(valid_probs) if value > neg_threshold]
        else:
            predict_error_index = [index for index, value in enumerate(valid_probs) if value == 1]
        return predict_error_index, val_acc



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--test_data', type=str, default=' ')
    args.add_argument('--start_end', type=str, default='start')
    args.add_argument('--seq_len', type=int, default=1500)
    args.add_argument('--parameter_choice', type=str, default='RU_2')
    args.add_argument('--batch_size', type=int, default=1024)
    args.add_argument('--dropout', type=float, default=0.1)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--show', action='store_true')
    args.add_argument('--save_path', type=str, default='./')
    args.add_argument('--dir_name', type=str, default='test')
    args.add_argument('--parameter_path', type=str, default='./')
    args.add_argument('--pos_neg', type=str, default='pos')
    args.add_argument('--device', type=str, default='cpu')

    args = args.parse_args()
    now = datetime.now()
    formatted_time_ = now.strftime("%m-%d-%H-%M-%S")
    args.dir_time = formatted_time_

    logger = utils.SafeLogger(args)
    logger.log(logging.INFO, f"Start Time & Fold Name: {formatted_time_}")
    utils.record_multi(logger, utils.record_parameter(args))

    logger.log(logging.INFO, 'The Testing is Starting')
    net = models.BreakModel(seq_len=args.seq_len, num_classes=1, parameter=args.parameter_choice,
                            dropout_rate=args.dropout)
    test_data = utils.load_single_data(args.test_data, pos_neg=args.pos_neg)
    test_dataset = utils.EccDNADataset(test_data, start_end=args.start_end)

    valid_model(net, test_dataset, args.parameter_path, threshold=args.threshold, batch_size=args.batch_size, device=args.device, args=args, logger=logger)
    logger.log(logging.INFO, 'The Testing is Finished !!!')



