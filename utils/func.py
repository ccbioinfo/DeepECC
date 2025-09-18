import os
import os.path
import logging

import pandas as pd
import torch
import queue
import threading
import numpy as np
from logging.handlers import RotatingFileHandler
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import matthews_corrcoef


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

_current_dir_name = 'default'
now = datetime.now()
formatted_time = now.strftime("%m-%d-%H-%M-%S")


class SafeLogger:
    def __init__(self, args):
        self.logger = logging.getLogger('SafeLogger')
        self.logger.setLevel(logging.INFO)

        log_dir = f'{args.save_path}/logs/{args.dir_name}'
        os.makedirs(log_dir, exist_ok=True)

        handler = RotatingFileHandler(
            filename=f'{log_dir}/metrics_{args.dir_time}.log',
            maxBytes=5 * 1024 * 1024,
        )

        formatter = logging.Formatter(
            fmt='%(levelname)s - %(asctime)s -- %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.log_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.log_thread = threading.Thread(
            target=self._log_worker,
            name='Logger',
            daemon=True
        )
        self.log_thread.start()

    def _log_worker(self):
        while not self.stop_event.is_set():
            try:
                record = self.log_queue.get(timeout=1)
                if record is not None:
                    self.logger.handle(record)
                self.log_queue.task_done()
            except queue.Empty:
                continue

    def log(self, level, msg, *args, **kwargs):
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=level,
            fn='',
            lno=0,
            msg=msg,
            args=args,
            exc_info=kwargs.get('exc_info', None),
            func=None,
            extra=None,
            sinfo=None
        )
        self.log_queue.put(record)

    def shutdown(self):
        self.stop_event.set()
        self.log_thread.join()

        while not self.log_queue.empty():
            record = self.log_queue.get()
            if record is not None:
                self.logger.handle(record)
            self.log_queue.task_done()


def record_multi(logger, content_list):
    for i in content_list:
        logger.log(logging.INFO, i)


def record_parameter(args):
    parameters = []

    for param in vars(args):
        parameters.append(f'{param}: {getattr(args, param)}')
    return parameters


def compute_pr_auc(true_label, predict_label):
    if np.all(true_label == 0) or np.all(true_label == 1):
        return 0.000
    precision, recall, _ = precision_recall_curve(true_label, predict_label)
    pr_auc = auc(recall, precision)
    return pr_auc


def compute_metrics(true_label, predict_label):

    acc = accuracy_score(true_label, predict_label)
    if all(label == 0 for label in true_label) or all(label == 1 for label in true_label):
        recall = 0.00
        precision = 0.00
        f1 = 0.00
        mcc = 0.00
    else:
        recall = recall_score(true_label, predict_label)
        precision = precision_score(true_label, predict_label)
        f1 = f1_score(true_label, predict_label)
        mcc = matthews_corrcoef(true_label, predict_label)
    return acc, recall, precision, f1, mcc


def show_metrics(epoch, train_loss, valid_loss, train_true_label, train_predict_label, valid_true_label, valid_predict_label):

    train_true_label, train_predict_label, valid_true_label, valid_predict_label = [
        t.cpu().numpy() if torch.is_tensor(t) else t
        for t in (train_true_label, train_predict_label, valid_true_label, valid_predict_label)
    ]

    train_acc, train_recall, train_pr, train_f1, train_mcc = compute_metrics(train_true_label, train_predict_label)
    valid_acc, valid_recall, valid_pr, valid_f1, valid_mcc = compute_metrics(valid_true_label, valid_predict_label)

    train_auc = compute_pr_auc(train_true_label, train_predict_label)
    valid_auc = compute_pr_auc(valid_true_label, valid_predict_label)

    log_content = (
        f'EPOCh: {epoch}',
        f'Train Loss: {train_loss:.4f}',
        f'Valid Loss: {valid_loss:.4f}',
        f'---{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    )
    log_content_1 = (
        f'Train: Acc-Recall-PR-F1-pr_auc-Mcc',
        f'{train_acc:.4f}-{train_recall:.4f}-{train_pr:.4f}-{train_f1:.4f}-{train_auc:.4f}-{train_mcc:.4f}'
    )
    log_content_2 = (
        f'Valid: Acc-Recall-PR-F1-pr_auc-Mcc',
        f'{valid_acc:.4f}-{valid_recall:.4f}-{valid_pr:.4f}-{valid_f1:.4f}-{valid_auc:.4f}-{valid_mcc:.4f}'
    )

    contents = [log_content, log_content_1, log_content_2]
    return contents


def show_metrics_valid(val_acc, val_recall, val_precision, val_f1, val_mcc):

    log_content = (
        f'Val Acc: {val_acc:.4f}',
        f'Val Recall: {val_recall:.4f}',
        f'Val PR: {val_precision:.4f}',
        f'Val F1: {val_f1:.4f}',
        f'Val Mcc: {val_mcc:.4f}'
    )
    return log_content


def save_prob_records(epoch, train_true_label, train_prob, valid_true_label, valid_prob, threshold,
                           save_path=None, dir_name=None, dir_time=None):
    train_true_label, train_prob, valid_true_label, valid_prob = [
        t.cpu().numpy() if torch.is_tensor(t) else t
        for t in (train_true_label, train_prob, valid_true_label, valid_prob)
    ]
    train_true_label = train_true_label.astype(int)
    valid_true_label = valid_true_label.astype(int)


    if dir_time is None:
        save_record = f'./Test_/records/{_current_dir_name}/{formatted_time}'
    else:
        save_record = f'{save_path}/records/{dir_name}/{dir_time}'
    os.makedirs(save_record, exist_ok=True)
    prob_df_train = pd.DataFrame({
        'train_true_label': train_true_label,
        'train_prob': train_prob,
    })
    prob_df_valid = pd.DataFrame({
        'valid_true_label': valid_true_label,
        'valid_prob': valid_prob
    })
    prob_df_train.to_csv(f'{save_record}/train_probs_result.csv', index=False)
    prob_df_valid.to_csv(f'{save_record}/valid_probs_result.csv', index=False)


def save_prob_records_(fold, train_true_label, train_prob, valid_true_label, valid_prob, threshold,
                           save_path=None, dir_name=None, dir_time=None):
    train_true_label, train_prob, valid_true_label, valid_prob = [
        t.cpu().numpy() if torch.is_tensor(t) else t
        for t in (train_true_label, train_prob, valid_true_label, valid_prob)
    ]
    train_true_label = train_true_label.astype(int)
    valid_true_label = valid_true_label.astype(int)


    if dir_time is None:
        save_record = f'./Test_/records/{_current_dir_name}/{formatted_time}'
    else:
        save_record = f'{save_path}/records/{dir_name}/{dir_time}'
    os.makedirs(save_record, exist_ok=True)
    prob_df_train = pd.DataFrame({
        'train_true_label': train_true_label,
        'train_prob': train_prob,
    })
    prob_df_valid = pd.DataFrame({
        'valid_true_label': valid_true_label,
        'valid_prob': valid_prob
    })
    prob_df_train.to_csv(f'{save_record}/train_probs_result_{fold}.csv', index=False)
    prob_df_valid.to_csv(f'{save_record}/valid_probs_result_{fold}.csv', index=False)



def save_prob_distribution(epoch, train_true_label, train_prob, valid_true_label, valid_prob, threshold,
                           save_path=None, dir_name=None, dir_time=None):

    train_true_label, train_prob, valid_true_label, valid_prob = [
        t.cpu().numpy() if torch.is_tensor(t) else t
        for t in (train_true_label, train_prob, valid_true_label, valid_prob)
    ]

    train_true_label = np.array(train_true_label).astype(int)
    train_prob = np.array(train_prob)

    if dir_time is None:
        save_dir = f'./Test_/plot_results/{_current_dir_name}/{formatted_time}'
    else:
        save_dir = f'{save_path}/plot_results/{dir_name}/{dir_time}'
    os.makedirs(save_dir, exist_ok=True)

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

    train_fpr, train_tpr, train_thresholds = roc_curve(train_true_label, train_prob)
    valid_fpr, valid_tpr, valid_thresholds = roc_curve(valid_true_label, valid_prob)
    train_roc_auc = auc(train_fpr, train_tpr)
    valid_roc_auc = auc(valid_fpr, valid_tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(train_fpr, train_tpr, label=f'train (auc = {train_roc_auc:.2f})', color='blue')
    plt.plot(valid_fpr, valid_tpr, label=f'valid (auc = {valid_roc_auc:.2f})', color='orange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.savefig(f'{save_dir}/ROC_Curve_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_prob_distribution_valid(valid_true_label, valid_prob, threshold,
                                 save_path=None, dir_name=None, dir_time=None):

    if dir_time is None:
        save_test_dir = f'./Test_/plot_results/{_current_dir_name}/Test/{formatted_time}'
    else:
        save_test_dir = f'{save_path}/plot_results/{dir_name}/Test/{dir_time}'
    # if not os.path.exists(save_test_dir):
    #     os.makedirs(save_test_dir)
    os.makedirs(save_test_dir, exist_ok=True)

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

    plt_save = f'{save_test_dir}/test_prob_{datetime.now().strftime("%m-%d-%H-%M-%S")}.png'
    # if os.path.exists(plt_save):
    #     plt_save = f'{save_test_dir}/test_prob_{threshold}_.png'

    plt.savefig(plt_save, dpi=300, bbox_inches='tight')
    plt.close()


def save_weights(logger, best_model_state_dict, save_path=None, dir_name=None, dir_time=None):
    if save_path is None:
        weight_path = f'./Test_/check_points/{_current_dir_name}'
    else:
        weight_path = f'{save_path}/check_points/{dir_name}'

    # if not os.path.exists(weight_path):
    #     os.makedirs(weight_path)
    os.makedirs(weight_path, exist_ok=True)
    if dir_time is None:
        torch.save(best_model_state_dict, os.path.join(weight_path, f'{formatted_time}.pth'))
    else:
        torch.save(best_model_state_dict, os.path.join(weight_path, f'{dir_time}.pth'))
    save_alert = f"Best model saved to {weight_path}"
    logger.log(logging.INFO, save_alert)


def get_save_weights(save_path=None, dir_name=None, dir_time=None):
    if save_path == None:
        weight_path = f'./Test_/check_points/{_current_dir_name}/{formatted_time}.pth'
    else:
        weight_path = f'{save_path}/check_points/{dir_name}/{dir_time}.pth'
    return weight_path


def record_end(logger, num):
    end_alert = f'The Best Epoch is:  {num}'
    end_alert_ = f'The Training is Finished'
    logger.log(logging.INFO, end_alert)
    logger.log(logging.INFO, end_alert_)


def is_smaller_than_all(number, lst):
    return all(number < x for x in lst)


def need_metric(train_acc, train_recall, train_pr, train_f1, val_acc):
    if train_acc > 0.7 and val_acc >= 0.99 and train_recall > 0.7 and train_pr > 0.7 and train_f1 > 0.7:
        return True
    else:
        return False


def find_parquet_files(root_dir):
    parquet_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    return parquet_files


normal_order = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
                     'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                     'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']

def combined_all(file_paths):
    all_sub_dfs = []

    for file_path in file_paths:

        file_name = os.path.basename(file_path)
        chr_ = file_name.split('_')[1]

        file_df = pd.read_parquet(file_path)
        file_df['site'] = np.arange(len(file_df))
        file_df = file_df[file_df['probs'] > 0.95]

        site_hg38 = file_df['site'] + 750
        chr_hg38 = np.full(len(file_df), chr_)

        sub_df = pd.DataFrame({
            'chr_hg38': chr_hg38,
            'site_hg38': site_hg38,
            'probs': file_df['probs'].values  
        })

        all_sub_dfs.append(sub_df)

    new_df = pd.concat(all_sub_dfs, axis=0, ignore_index=True)
    
    new_df['chr_hg38'] = pd.Categorical(
        new_df['chr_hg38'],
        categories=normal_order,
        ordered=True
    )
    new_df = new_df.sort_values(['chr_hg38', 'site_hg38'])
    new_df = new_df.reset_index(drop=True)

    return new_df

