import multiprocessing
import os
import os.path
import logging
import sys
from logging.handlers import QueueHandler, QueueListener
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

now = datetime.now()
formatted_time = now.strftime("%m-%d-%H-%M-%S")

_current_dir_name = 'default'
log_queue = None
_queue_listener = None

def setup_logging(dir_name, time=None):
    global _current_dir_name
    _current_dir_name = dir_name
    log_dir = f'/data01/home/wangchangcheng/EccDNA_project_mini/Test_/logs/{_current_dir_name}'
    os.makedirs(log_dir, exist_ok=True)
    if time is None:
        log_filename = os.path.join(log_dir, f'metrics_{formatted_time}.log')
    else:
        log_filename = os.path.join(log_dir, f'metrics_{time}.log')
    logging.basicConfig(
        level=logging.INFO,
        filename=log_filename,
        filemode='a',
        format='%(levelname)s - %(asctime)s -- %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def record_content(content):
    logging.info(content)


def record_parameter(args):
    pos_data_info = f'pos_data: {args.pos_data}'
    logging.info(pos_data_info)
    neg_data_info = f'neg_data: {args.neg_data}'
    logging.info(neg_data_info)
    start_end_info = f'start_end: {args.start_end}'
    logging.info(start_end_info)
    batch_size_info = f'batch_size: {args.batch_size}'
    logging.info(batch_size_info)
    lr_info = f'lr: {args.lr}'
    logging.info(lr_info)
    wd_info = f'wd: {args.wd}'
    logging.info(wd_info)
    smoothing_info = f'smoothing: {args.smoothing}'
    logging.info(smoothing_info)
    threshold_info = f'threshold: {args.threshold}'
    logging.info(threshold_info)
    val_split_info = f'val_split: {args.val_split}'
    logging.info(val_split_info)
    epochs_info = f'epochs: {args.epochs}'
    logging.info(epochs_info)
    save_info = f'save: {args.save}'
    logging.info(save_info)
    show_info = f'show: {args.show}'
    logging.info(show_info)
    device_info = f'device: {args.device}'
    logging.info(device_info)
    seq_len_info = f'seq_len: {args.seq_len}'
    logging.info(seq_len_info)
    parameter_choice_info = f'parameter_choice: {args.parameter_choice}'
    logging.info(parameter_choice_info)
    dropout_info = f'dropout: {args.dropout}'
    logging.info(dropout_info)
    weight_info = f'weight: {args.pos_weight}'
    logging.info(weight_info)


def compute_pr_auc(true_label, predict_label):
    precision, recall, _ = precision_recall_curve(true_label, predict_label)
    pr_auc = auc(recall, precision)
    return pr_auc


def compute_metrics(true_label, predict_label):

    acc = accuracy_score(true_label, predict_label)
    recall = recall_score(true_label, predict_label)
    precision = precision_score(true_label, predict_label)
    f1 = f1_score(true_label, predict_label)

    return acc, recall, precision, f1


def show_metrics(epoch, train_loss, valid_loss, train_true_label, train_predict_label, valid_true_label, valid_predict_label):
    train_acc, train_recall, train_pr, train_f1 = compute_metrics(train_true_label, train_predict_label)
    valid_acc, valid_recall, valid_pr, valid_f1 = compute_metrics(valid_true_label, valid_predict_label)
    train_auc = compute_pr_auc(train_true_label, train_predict_label)
    valid_auc = compute_pr_auc(valid_true_label, valid_predict_label)

    log_content = (
        f'EPOCh: {epoch}',
        f'Train Loss: {train_loss:.4f}',
        f'Valid Loss: {valid_loss:.4f}',
        f'---{datetime.now()}'
    )
    log_content_1 = (
        f'Train Acc-Train Recall-Train PR-Train F1-Train AUC',
        f'{train_acc:.4f}-{train_recall:.4f}-{train_pr:.4f}-{train_f1:.4f}-{train_auc:.4f}'
    )
    log_content_2 = (
        f'Valid Acc-Valid Recall-Valid PR-Valid F1-Valid AUC',
        f'{valid_acc:.4f}-{valid_recall:.4f}-{valid_pr:.4f}-{valid_f1:.4f}-{valid_auc:.4f}'
    )

    logging.info(log_content)
    logging.info(log_content_1)
    logging.info(log_content_2)


def show_metrics_parallel(epoch, train_loss, valid_loss, train_true_label, train_predict_label, valid_true_label, valid_predict_label):

    train_true_label, train_predict_label, valid_true_label, valid_predict_label = [
        t.cpu().numpy() if torch.is_tensor(t) else t
        for t in (train_true_label, train_predict_label, valid_true_label, valid_predict_label)
    ]

    train_acc, train_recall, train_pr, train_f1 = compute_metrics(train_true_label, train_predict_label)
    valid_acc, valid_recall, valid_pr, valid_f1 = compute_metrics(valid_true_label, valid_predict_label)
    train_auc = valid_auc = 0.0001

    log_content = (
        f'EPOCh: {epoch}',
        f'Train Loss: {train_loss:.4f}',
        f'Valid Loss: {valid_loss:.4f}',
        f'---{datetime.now()}'
    )
    log_content_1 = (
        f'Train Acc-Train Recall-Train PR-Train F1-Train AUC',
        f'{train_acc:.4f}-{train_recall:.4f}-{train_pr:.4f}-{train_f1:.4f}-{train_auc:.4f}'
    )
    log_content_2 = (
        f'Valid Acc-Valid Recall-Valid PR-Valid F1-Valid AUC',
        f'{valid_acc:.4f}-{valid_recall:.4f}-{valid_pr:.4f}-{valid_f1:.4f}-{valid_auc:.4f}'
    )

    logging.info(log_content)
    logging.info(log_content_1)
    logging.info(log_content_2)


def show_metrics_valid(val_acc, val_recall, val_precision, val_f1):

    log_content = (
        f'Val Acc: {val_acc:.4f}',
        f'Val Recall: {val_recall:.4f}',
        f'Val PR: {val_precision:.4f}',
        f'Val F1: {val_f1:.4f}',
    )
    logging.info(log_content)


def save_prob_distribution(epoch, train_true_label, train_prob, valid_true_label, valid_prob, threshold, time=None):

    train_true_label, train_prob, valid_true_label, valid_prob = [
        t.cpu().numpy() if torch.is_tensor(t) else t
        for t in (train_true_label, train_prob, valid_true_label, valid_prob)
    ]

    train_true_label = np.array(train_true_label).astype(int)
    train_prob = np.array(train_prob)

    if time is None:
        save_dir = f'/data01/home/wangchangcheng/EccDNA_project_mini/Test_/plot_results/{_current_dir_name}/{formatted_time}'
    else:
        save_dir = f'/data01/home/wangchangcheng/EccDNA_project_mini/Test_/plot_results/{_current_dir_name}/{time}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        creat_alert = f"'{formatted_time}' created !!!"
        logging.info(creat_alert)

    plt.figure(figsize=(8, 6))
    sns.histplot(train_prob[train_true_label == 0], color='blue', label='Class 0', kde=True, bins=100, alpha=0.5, stat='probability')
    sns.histplot(train_prob[train_true_label == 1], color='orange', label='Class 1', kde=True, bins=100, alpha=0.5, stat='probability')
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5, label='Threshold')
    plt.axhline(y=0.01, color='green', linestyle=':', linewidth=1.5, label='y=0.01')
    plt.axhline(y=0.005, color='green', linestyle=':', linewidth=1.5, label='y=0.005')
    plt.title('Probability Distribution by True Class (Train)', fontsize=16)
    plt.xlabel('Predicted Probability', fontsize=14)
    plt.ylabel('Frequency/Density', fontsize=14)
    plt.legend(title='True Class')
    plt.grid(axis='y', linestyle='--', alpha=0.7)


    plt.savefig(f'{save_dir}/train_prob_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

    valid_true_label = np.array(valid_true_label).astype(int)
    valid_prob = np.array(valid_prob)

    plt.figure(figsize=(8, 6))
    sns.histplot(valid_prob[valid_true_label == 0], color='blue', label='Class 0', kde=True, bins=100, alpha=0.5, stat='probability')
    sns.histplot(valid_prob[valid_true_label == 1], color='orange', label='Class 1', kde=True, bins=100, alpha=0.5, stat='probability')
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5, label='Threshold')
    plt.axhline(y=0.01, color='green', linestyle=':', linewidth=1.5, label='y=0.01')
    plt.axhline(y=0.005, color='green', linestyle=':', linewidth=1.5, label='y=0.005')
    plt.title('Probability Distribution by True Class (Valid)', fontsize=16)
    plt.xlabel('Predicted Probability', fontsize=14)
    plt.ylabel('Frequency/Density', fontsize=14)
    plt.legend(title='True Class')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(f'{save_dir}/valid_prob_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_prob_distribution_valid(valid_true_label, valid_prob, threshold, time=None):
    if time is None:
        save_test_dir = f'/data01/home/wangchangcheng/EccDNA_project_mini/Test_/plot_results/{_current_dir_name}/Test/{formatted_time}'
    else:
        save_test_dir = f'/data01/home/wangchangcheng/EccDNA_project_mini/Test_/plot_results/{_current_dir_name}/Test/{time}'
    if not os.path.exists(save_test_dir):
        os.makedirs(save_test_dir)

    valid_true_label = np.array(valid_true_label).astype(int)
    valid_prob = np.array(valid_prob)

    plt.figure(figsize=(8, 6))
    sns.histplot(valid_prob[valid_true_label == 0], color='blue', label='Class 0', kde=True, bins=100, alpha=0.5, stat='probability')
    sns.histplot(valid_prob[valid_true_label == 1], color='orange', label='Class 1', kde=True, bins=100, alpha=0.5, stat='probability')
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5, label='Threshold')
    plt.axhline(y=0.01, color='green', linestyle=':', linewidth=1.5, label='y=0.01')
    plt.axhline(y=0.005, color='green', linestyle=':', linewidth=1.5, label='y=0.005')
    plt.title('Probability Distribution by True Class (Valid)', fontsize=16)
    plt.xlabel('Predicted Probability', fontsize=14)
    plt.ylabel('Frequency/Density', fontsize=14)
    plt.legend(title='True Class')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt_save = f'{save_test_dir}/test_prob_{threshold}.png'
    if os.path.exists(plt_save):
        plt_save = f'{save_test_dir}/test_prob_{threshold}_.png'

    plt.savefig(plt_save, dpi=300, bbox_inches='tight')
    plt.close()

    finish_alert = f'The Folder Name is {formatted_time}'
    logging.info(finish_alert)


def save_weights(best_model_state_dict, time=None):
    save_path = f'/data01/home/wangchangcheng/EccDNA_project_mini/Test_/check_points/{_current_dir_name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if time is None:
        torch.save(best_model_state_dict, os.path.join(save_path, f'{formatted_time}.pth'))
    else:
        torch.save(best_model_state_dict, os.path.join(save_path, f'{time}.pth'))
    save_alert = f"Best model saved to {save_path}"
    logging.info(save_alert)


def get_save_weights(dir_name=None, time=None):
    if dir_name is None or time is None:
        save_path = f'/data01/home/wangchangcheng/EccDNA_project_mini/Test_/check_points/{_current_dir_name}/{formatted_time}.pth'
    else:
        save_path = f'/data01/home/wangchangcheng/EccDNA_project_mini/Test_/check_points/{dir_name}/{time}.pth'
    return save_path


def record_end(num):
    end_alert = f'The Best Epoch is:  {num}'
    end_alert_ = f'The Training is Finished'
    logging.info(end_alert)
    logging.info(end_alert_)


def is_smaller_than_all(number, lst):
    return all(number < x for x in lst)


def need_metric(train_acc, train_recall, train_pr, train_f1, val_acc):
    if train_acc > 0.7 and val_acc >= 0.99 and train_recall > 0.7 and train_pr > 0.7 and train_f1 > 0.7:
        return True
    else:
        return False


if __name__ == '__main__':
    train_true_label = [1, 1, 0, 0, 1]
    train_predict_label = [1, 1, 0, 0, 0]
    valid_true_label = [1, 1, 0, 0, 1]
    valid_predict_label = [1, 1, 0, 0, 0]
    show_metrics(1, 0.0001, 0.0001, train_true_label, train_predict_label, valid_true_label, valid_predict_label)








