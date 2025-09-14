import argparse

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

import utils
import models
from models import *
import valid
import datetime


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def train_model(model, dataset, epochs, lr=1e-3, wd=0, threshold=0.5, val_split=0.2, smoothing=0.1, batch_size=10240,
                 show=True, device='cpu', pos_weight=None, parallel=False, logger=None):

    if parallel:
        device_ids = list(range(torch.cuda.device_count()))
        model = DataParallel(model, device_ids=device_ids).to(device_ids[0])
        device = device_ids[0]
    else:
        model.to(device)
    if pos_weight is not None:
        pos_weight = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=16, pin_memory=True, persistent_workers=True)

    best_val_loss = [1000]
    best_model_state_dict = None
    best_epochs = []

    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_true_label = []
        train_predict_label = []
        train_probs = []

        for input, label in train_loader:
            input = input.to(torch.float32).to(device)
            smoothing_label = label * (1 - smoothing) + (1 - label) * smoothing
            label = label.to(torch.float32).to(device)
            smoothing_label = smoothing_label.to(torch.float32).to(device)
            output = model(input)

            optimizer.zero_grad()
            loss = criterion(output, smoothing_label)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            prob = torch.sigmoid(output)
            train_probs.extend(prob.tolist())
            temp_predict = (prob > threshold)
            train_predict_label.extend(temp_predict.tolist())
            train_true_label.extend(label.tolist())

        avg_train_loss = sum(train_loss) / len(train_loss)

        model.eval()
        val_loss = []
        valid_true_label = []
        valid_predict_label = []
        valid_probs = []
        with torch.no_grad():
            for input, label in val_loader:
                input = input.to(torch.float32).to(device)
                smoothing_label = label * (1 - smoothing) + (1 - label) * smoothing
                label = label.to(torch.float32).to(device)
                smoothing_label = smoothing_label.to(torch.float32).to(device)
                output = model(input)

                loss = criterion(output, smoothing_label)
                val_loss.append(loss.item())

                prob = torch.sigmoid(output)
                valid_probs.extend(prob.tolist())
                temp_predict_ = (prob > threshold)
                valid_predict_label.extend(temp_predict_.tolist())
                valid_true_label.extend(label.tolist())

        avg_val_loss = sum(val_loss) / len(val_loss)
        valid_acc, valid_recall, valid_pr, valid_f1, valid_mcc = compute_metrics(valid_true_label, valid_predict_label)

        metrics_contents = show_metrics(epoch, avg_train_loss, avg_val_loss, train_true_label, train_predict_label, valid_true_label, valid_predict_label)
        utils.record_multi(logger, metrics_contents)
        if show:
            save_prob_distribution(epoch, train_true_label, train_probs, valid_true_label, valid_probs, threshold,
                                   save_path=args.save_path, dir_name=args.dir_name, dir_time=args.dir_time)

        if is_smaller_than_all(avg_val_loss, best_val_loss):
            if parallel:
                best_model_state_dict = model.module.state_dict()
            else:
                best_model_state_dict = model.state_dict()
            best_epochs.append(epoch)
        best_val_loss.append(avg_val_loss)


    save_weights(logger, best_model_state_dict,
                     save_path=args.save_path, dir_name=args.dir_name, dir_time=args.dir_time)

    utils.record_end(logger, best_epochs[-1])


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--pos_data', type=str, default=' ')
    args.add_argument('--neg_data', type=str, default=' ')
    args.add_argument('--start_end', type=str, default='start')
    args.add_argument('--seq_len', type=int, default=1500)
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
    args.add_argument('--parallel', action='store_true')
    args.add_argument('--device', type=str, default='cpu')


    args = args.parse_args()
    now = datetime.datetime.now()
    formatted_time_ = now.strftime("%m-%d-%H-%M-%S")
    args.dir_time = formatted_time_

    logger = utils.SafeLogger(args)
    logger.log(logging.INFO, f"Start Time & Fold Name: {formatted_time_}")
    utils.record_multi(logger, utils.record_parameter(args))

    train_data = utils.load_data(args.pos_data, args.neg_data, shuffled=True)
    train_dataset = utils.EccDNADataset(train_data, start_end=args.start_end)
    net = models.BreakModel(seq_len=args.seq_len, num_classes=1, parameter=args.parameter_choice, dropout_rate=args.dropout)
    train_model(net, train_dataset, epochs=args.epochs, lr=args.lr, wd=args.wd, smoothing=args.smoothing, threshold=args.threshold,
                val_split=args.val_split, batch_size=args.batch_size, show=args.show, device=args.device,
                parallel=args.parallel, logger=logger)
















