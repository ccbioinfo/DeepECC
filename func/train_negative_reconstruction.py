import argparse
import os.path

import utils
import models
from models import *
import train
import valid

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def train_model_negative_reconstruction(args):
    train_data = utils.load_data(args.pos_data, args.neg_data, shuffled=True)
    train_dataset = utils.EccDNADataset(train_data, start_end=args.start_end)
    net = models.BreakModel(seq_len=args.seq_len, num_classes=1, parameter=args.parameter_choice,
                            dropout_rate=args.dropout)

    test_data_file = args.neg_file_list
    for i in range(10):
        train_acc, train_recall, train_pr, train_f1 = train.train_model(net, train_dataset, epochs=args.epochs, lr=args.lr, wd=args.wd, smoothing=args.smoothing, threshold=args.threshold,
                                                                        val_split=args.val_split, batch_size=args.batch_size, save=args.save, show=args.show, device=args.device,
                                                                        df=True, parallel=args.parallel)

        parameter_path = get_save_weights()

        test_data = utils.load_single_data(test_data_file[i], pos_neg='neg')
        test_dataset = utils.EccDNADataset(test_data, start_end=args.start_end)
        error_index, val_acc = valid.valid_model(net, test_dataset, parameter_path, threshold=0.5, batch_size=1024,
                                                 device=args.device, df=True, parallel=args.parallel)
        test_data_error = test_data.iloc[error_index]

        if not need_metric(train_acc, train_recall, train_pr, train_f1, val_acc):
            train_data = pd.concat([train_data, test_data_error], axis=0)
            train_dataset = utils.EccDNADataset(train_data, start_end=args.start_end)
        else:
            break
        record_end(i)
    record_end(i)


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--pos_data', type=str, default=' ')
    args.add_argument('--neg_data', type=str, default=' ')
    args.add_argument('--neg_file_list', type=str, default=' ')
    args.add_argument('--start_end', type=str, default='start')
    args.add_argument('--seq_len', type=int, default=1500)
    args.add_argument('--parameter_choice', type=str, default='RU_2')
    args.add_argument('--epochs', type=int, default=30)
    args.add_argument('--batch_size', type=int, default=10240)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--wd', type=float, default=0)
    args.add_argument('--pos_weight', type=float, default=None)
    args.add_argument('--smoothing', type=float, default=0.1)
    args.add_argument('--dropout', type=float, default=0.2)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--val_split', type=float, default=0.2)
    args.add_argument('--save', action='store_true')
    args.add_argument('--show', action='store_true')
    args.add_argument('--device', type=str, default='cpu')


    args = args.parse_args()
    record_parameter(args)
    train_model_negative_reconstruction(args)























