import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool

chr_length = {
    'chr1':248956422, 'chr2':242193529, 'chr3':198295559, 'chr4':190214555, 'chr5':181538259,
    'chr6':170805979, 'chr7':159345973, 'chr8':145138636, 'chr9':138394717, 'chr10':133797422,
    'chr11':135086622, 'chr12':133275309, 'chr13':114364328, 'chr14':107043718, 'chr15':101991189,
    'chr16':90338345, 'chr17':83257441, 'chr18':80373285, 'chr19':58617616, 'chr20':64444167, 'chr21':46709983,
    'chr22':50818468, 'chrX':156040895, 'chrY':57227415
}

df_start_dict = {}
df_end_dict = {}

def load_chr_df(path, chrom):
    df = pd.read_parquet(path)
    df.columns = ['chr', 'point', 'value']
    df = df[df['chr'] == chrom]
    df_array = df['point'].values
    df_array = df_array.astype(np.int32)
    return df_array


def joint_data_generate(chromosome, start_path, end_path, save_dir, min_length, max_length):
    print(datetime.now().strftime("%m-%d-%H-%M-%S"))
    print(f'Prepare {chromosome} joint data start >>>')

    df_start_ = load_chr_df(start_path, chromosome)
    df_end_ = load_chr_df(end_path, chromosome)

    if df_start_.size == 0 or df_end_.size == 0:
        print(f"{chromosome} has no data, skipped.")
        return

    child_dir = os.path.join(save_dir, chromosome)
    os.makedirs(child_dir, exist_ok=True)

    batch_size = 100000000
    batch = []
    file_index = 0

    for start_point in df_start_:
        min_start = start_point + min_length
        max_start = min(start_point + max_length, chr_length[chromosome])
        if min_start >= max_start:
            continue

        mask = (df_end_ < max_start) & (df_end_ > min_start)
        matched_end = df_end_[mask]

        if matched_end.size == 0:
            continue

        batch.extend([(chromosome, start_point, e) for e in matched_end])

        if len(batch) >= batch_size:
            tmp_file = os.path.join(child_dir, f"{chromosome}_part_{file_index}.parquet")
            pd.DataFrame(batch, columns=['chr', 'start', 'end']).to_parquet(tmp_file, compression='snappy')
            file_index += 1
            batch.clear()

    if batch:
        tmp_file = os.path.join(child_dir, f"{chromosome}_part_{file_index}.parquet")
        pd.DataFrame(batch, columns=['chr', 'start', 'end']).to_parquet(tmp_file, compression='snappy')
        batch.clear()

    print(datetime.now().strftime("%m-%d-%H-%M-%S"))
    print(f'Prepare {chromosome} joint data fulfill <<<')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--start_path', type=str, required=True)
    args.add_argument('--end_path', type=str, required=True)
    args.add_argument('--save_dir', type=str, required=True)
    args.add_argument('--min_length', type=int, default=50)
    args.add_argument('--max_length', type=int, default=500000)
    args.add_argument('--multi_core', type=int, default=8)

    args = args.parse_args()

    print(datetime.now().strftime("%m-%d-%H-%M-%S"))
    print(f'Prepare Joint data Start...')

    os.makedirs(args.save_dir, exist_ok=True)
    chromosome_list = list(chr_length.keys())

    with Pool(processes=args.multi_core) as pool:
        pool.starmap(joint_data_generate, [
            (chrom, args.start_path, args.end_path, args.save_dir, args.min_length, args.max_length)
            for chrom in chromosome_list
        ])

    print(datetime.now().strftime("%m-%d-%H-%M-%S"))
    print(f'Prepare Joint data Finish!!!')



