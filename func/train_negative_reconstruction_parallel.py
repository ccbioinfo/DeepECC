import shutil
import pandas as pd
import glob
import time

from train_parallel import *

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def glob_list(path):
    matching_files = glob.glob(f'{path}*')
    matching_files = matching_files[1:]
    return matching_files


def copy_pth(original_file, fold):
    file_dir = os.path.dirname(original_file)
    file_name, file_ext = os.path.splitext(os.path.basename(original_file))
    new_file_name = f"{file_name}_{fold}{file_ext}"
    new_file_path = os.path.join(file_dir, new_file_name)
    shutil.copy2(original_file, new_file_path)


def main(args, neg_file_list, logger):

    for i in range(len(neg_file_list)):
        mp.spawn(main_worker,
                 args=(args.world_size, args),
                 nprocs=args.world_size,
                 join=True)
        net = models.BreakModel(seq_len=args.seq_len, num_classes=1, parameter=args.parameter_choice,
                                dropout_rate=args.dropout)
        parameter_path = utils.get_save_weights(save_path=args.save_path, dir_name=args.dir_name, dir_time=args.dir_time)
        copy_pth(parameter_path, i)

        test_data = utils.load_single_data(neg_file_list[i], pos_neg='neg')
        test_dataset = utils.EccDNADataset(test_data, start_end=args.start_end)
        error_index, val_acc = valid.valid_model(net, test_dataset, parameter_path, threshold=0.5, batch_size=1024, device='cuda:2',
                                                 args=args, logger=logger, df=True, neg_threshold=args.neg_threshold)
        rate = round(len(error_index) / len(test_data), 4)
        logger.log(logging.INFO, f'The new negative dataset valid rate is {rate}, len is {len(error_index)}')

        if rate <= 0.02:
            logger.log(logging.INFO, 'Reach the setting condition')
            break
        test_data_error = test_data.iloc[error_index]
        original_neg_data = utils.load_single_data(args.neg_data, pos_neg='neg')
        temp_data = pd.concat([original_neg_data, test_data_error], axis=0)
        temp_path = f'./{args.dir_time}_temp_data.csv'
        temp_data.to_csv(temp_path, index=False, header=True)
        args.neg_data = temp_path
        time.sleep(30)
        logger.log(logging.INFO, f'Finish the {i} training')



if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--pos_data', type=str, default=' ')
    args.add_argument('--neg_data', type=str, default=' ')
    args.add_argument('--neg_file_list', type=str, default=None)
    args.add_argument('--neg_threshold', type=float, default=0.5)
    args.add_argument('--start_end', type=str, default='start')
    args.add_argument('--seq_len', type=int, default=1500)
    args.add_argument('--joint', action='store_true')
    args.add_argument('--parameter_choice', type=str, default='RU_2')
    args.add_argument('--epochs', type=int, default=200)
    args.add_argument('--batch_size', type=int, default=10240)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--wd', type=float, default=0)
    args.add_argument('--pos_weight', type=float, default=None)
    args.add_argument('--dropout', type=float, default=0.1)
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
    neg_file_list = glob_list(args.neg_file_list)

    logger = utils.SafeLogger(args)
    logger.log(logging.INFO, f'Start Time & Fold Name: {formatted_time_}')
    utils.record_multi(logger, utils.record_parameter(args))

    main(args, neg_file_list, logger)

    logger.log(logging.INFO, 'The negative reconstruction is Finished !!!')



