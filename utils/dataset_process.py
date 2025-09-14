import os
import shutil
import numpy as np
import pandas as pd
import random
import secrets
from Bio import SeqIO
from multiprocessing import Manager, Pool


normal_chromosome = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
                     'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                     'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY', 'chr23', 'chr24',
                     'chr25', 'chr26', 'chr27', 'chr28', 'chrW', 'chrZ']


standard_chromosome = ['CM000663.2', 'CM000664.2', 'CM000665.2', 'CM000666.2', 'CM000667.2', 'CM000668.2',
                       'CM000669.2', 'CM000670.2', 'CM000671.2', 'CM000672.2', 'CM000673.2', 'CM000674.2',
                       'CM000675.2', 'CM000676.2', 'CM000677.2', 'CM000678.2', 'CM000679.2', 'CM000680.2',
                       'CM000681.2', 'CM000682.2', 'CM000683.2', 'CM000684.2', 'CM000685.2', 'CM000686.2']

chromosome_length = {
    'chr1':248956422, 'chr2':242193529, 'chr3':198295559, 'chr4':190214555, 'chr5':181538259,
    'chr6':170805979, 'chr7':159345973, 'chr8':145138636, 'chr9':138394717, 'chr10':133797422,
    'chr11':135086622, 'chr12':133275309, 'chr13':114364328, 'chr14':107043718, 'chr15':101991189,
    'chr16':90338345, 'chr17':83257441, 'chr18':80373285, 'chr19':58617616, 'chr20':64444167, 'chr21':46709983,
    'chr22':50818468, 'chrX':156040895, 'chrY':57227415,
}

chromosome_length_mouse = {"chr1": 195471971, "chr2": 182113224, "chr3": 160039680, "chr4": 156508116, "chr5": 151834684,
                           "chr6": 149736546, "chr7": 145441459, "chr8": 129401213, "chr9": 124595110, "chr10": 130694993,
                           "chr11": 122082543, "chr12": 120129022, "chr13": 120421639, "chr14": 124902244, "chr15": 104043685,
                           "chr16": 98207768, "chr17": 94987271, "chr18": 90702639, "chr19": 61431566, "chrX": 171031299,
                           "chrY": 91744698}

chromosome_length_gallus = { "chr1": 196449156, "chr2": 149539284, "chr3": 110642502, "chr4": 90861225, "chr5": 59506338, "chr6": 36220557,
                             "chr7": 36382834, "chr8": 29578256, "chr9": 23733309, "chr10": 20453248, "chr11": 19638187, "chr12": 20119077,
                             "chr13": 17905061, "chr14": 15331188, "chr15": 12703657, "chr16": 2706039, "chr17": 11092391, "chr18": 11623896,
                             "chr19": 10455293, "chr20": 14265659, "chr21": 6970754, "chr22": 4686657, "chr23": 6253421, "chr24": 6478339,
                             "chr25": 3067737, "chr26": 5349051, "chr27": 5228753, "chr28": 5437364, "chr29": 726478, "chr30": 755666,
                             "chr31": 2457334, "chr32": 125424, "chr33": 3839931, "chr34": 3469343, "chr35": 554126, "chr36": 358375, "chr37": 157853,
                             "chr38": 667312, "chr39": 177356, "chrW": 9109940, "chrZ": 86044486}


random.seed(42)
np.random.seed(42)
rng = np.random.default_rng(seed=42)


def data_preprocess(seq_data, genome_file, around_len=1024, read_data=True, filtering=False, species='human'):

    around_len = int(around_len // 2)

    if read_data:
        try:
            if seq_data.endswith('.csv'):
                eccDNA_raw_data = pd.read_table(seq_data, sep=' ')
            else:
                eccDNA_raw_data = pd.read_table(seq_data)
        except:
            try:
                eccDNA_raw_data = pd.read_table(seq_data, sep=' ')
            except:
                eccDNA_raw_data = pd.read_table(seq_data, sep=' ', dtype={13: str})
    else:
        eccDNA_raw_data = seq_data

    if filtering:
        eccDNA_raw_data = eccDNA_raw_data[eccDNA_raw_data['Length'] > 1e5]
    eccDNA_raw_data = eccDNA_raw_data[['eccid', 'chr_hg38', 'start_hg38', 'end_hg38', 'Length']]

    eccDNA_raw_data = eccDNA_raw_data.dropna()

    eccDNA_raw_data['start_hg38'] = eccDNA_raw_data['start_hg38'].astype(int)
    eccDNA_raw_data['end_hg38'] = eccDNA_raw_data['end_hg38'].astype(int)

    eccDNA_raw_data = eccDNA_raw_data[eccDNA_raw_data['chr_hg38'].isin(normal_chromosome)]
    eccDNA_raw_data = eccDNA_raw_data[eccDNA_raw_data['Length'] != 0]
    eccDNA_raw_data = eccDNA_raw_data[eccDNA_raw_data['start_hg38'] != eccDNA_raw_data['end_hg38']]
    eccDNA_raw_data = eccDNA_raw_data.drop_duplicates(subset=['chr_hg38', 'start_hg38'], keep='first')
    eccDNA_raw_data = eccDNA_raw_data.drop_duplicates(subset=['chr_hg38', 'end_hg38'], keep='first')

    if species == 'human':
        chr_to_ac = {key: value for key, value in zip(normal_chromosome, standard_chromosome)}
        eccDNA_raw_data['ac_number'] = eccDNA_raw_data['chr_hg38'].map(chr_to_ac)
    elif species == 'mouse':
        eccDNA_raw_data['ac_number'] = eccDNA_raw_data['chr_hg38']
    elif species == 'gallus':
        eccDNA_raw_data['ac_number'] = eccDNA_raw_data['chr_hg38'].str.replace('chr', '', regex=False)

    fasta_sequences = SeqIO.to_dict(SeqIO.parse(genome_file, "fasta"))
    fasta_keys = fasta_sequences.keys()
    around_start = []
    around_end = []
    for i in range(len(eccDNA_raw_data)):
        ac_number = eccDNA_raw_data['ac_number'].iloc[i]
        chr_number = eccDNA_raw_data['chr_hg38'].iloc[i]
        chr_number_ = chr_number[3:]
        start_position = eccDNA_raw_data['start_hg38'].iloc[i]
        end_position = eccDNA_raw_data['end_hg38'].iloc[i]

        if chr_number_ in fasta_keys:
            chr_sequence = fasta_sequences.get(chr_number_)
        elif chr_number in fasta_keys:
            chr_sequence = fasta_sequences.get(chr_number)
        elif ac_number in fasta_keys:
            chr_sequence = fasta_sequences.get(ac_number)
        else:
            raise KeyError(f'No such chromosome')

        if start_position > around_len:
            start_sub_sequence = chr_sequence.seq[(start_position - around_len):(start_position + around_len)]
        else:
            start_sub_sequence = chr_sequence.seq[start_position:(start_position + around_len*2)]
        start_sub_sequence = str(start_sub_sequence)
        start_sub_sequence = start_sub_sequence.upper()

        end_sub_sequence = chr_sequence.seq[(end_position - around_len):(end_position + around_len)]
        end_sub_sequence = str(end_sub_sequence)
        end_sub_sequence = end_sub_sequence.upper()

        around_start.append(start_sub_sequence)
        around_end.append(end_sub_sequence)

    eccDNA_raw_data['around_start'] = around_start
    eccDNA_raw_data['around_end'] = around_end
    eccDNA_raw_data[['around_start', 'around_end']] = eccDNA_raw_data[['around_start', 'around_end']].replace(r'^\s*$', np.nan, regex=True)
    eccDNA_raw_data = eccDNA_raw_data.dropna()

    eccDNA_raw_data = eccDNA_raw_data[eccDNA_raw_data['around_start'].str.len() == (around_len*2)]
    eccDNA_raw_data = eccDNA_raw_data[eccDNA_raw_data['around_end'].str.len() == (around_len*2)]
    return eccDNA_raw_data


def generate_single_false(index, data_frame, eccid_list,
                          chr_list, start_hg38_list, end_hg38_list,
                          length_list, around_len=3000, species='human', seed=42):

    np.random.seed(seed + index)
    eccid = data_frame['eccid'].iloc[index]
    chr = data_frame['chr_hg38'].iloc[index]
    ac = data_frame['ac_number'].iloc[index]
    length = data_frame['end_hg38'].iloc[index] - data_frame['start_hg38'].iloc[index]

    all_point_ = data_frame[data_frame['chr_hg38'] == chr]
    all_start_point = list(all_point_['start_hg38'])
    all_end_point = list(all_point_['end_hg38'])

    if species == 'mouse':
        chr_length = chromosome_length_mouse[chr]
    elif species == 'gallus':
        chr_length = chromosome_length_gallus[chr]
    else:
        chr_length = chromosome_length[chr]

    random_down = around_len
    half_around_len = around_len * 2
    random_upper = chr_length - length - around_len

    random_start_point = np.random.randint(random_down, random_upper)
    random_end_point = random_start_point + length

    while any(random_start_point - half_around_len <= x <= random_start_point + half_around_len for x in all_start_point) or \
            any(random_end_point - half_around_len <= x <= random_end_point + half_around_len for x in all_end_point):

        random_start_point = np.random.randint(random_down, random_upper)
        random_end_point = random_start_point + length

    eccid_list.append(eccid + '_false')
    chr_list.append(chr)
    start_hg38_list.append(random_start_point)
    end_hg38_list.append(random_end_point)
    length_list.append(length)


def generate_false_data(data_frame, around_len=3000, multi_process=False, species='human', seed=42):
    around_len = int(around_len // 2)
    data_frame = data_frame.sample(frac=1, random_state=seed).reset_index(drop=True)

    if multi_process:
        index_list = [i for i in range(len(data_frame))]

        with Manager() as manager:
            eccid_list = manager.list()
            chr_list = manager.list()
            start_hg38_list = manager.list()
            end_hg38_list = manager.list()
            length_list = manager.list()

            with Pool(processes=20) as p:
                p.starmap(generate_single_false,
                          [(index, data_frame, eccid_list,
                            chr_list, start_hg38_list, end_hg38_list, length_list,
                            around_len, species, seed) for index in index_list])

            eccid_list = list(eccid_list)
            chr_list = list(chr_list)
            start_hg38_list = list(start_hg38_list)
            end_hg38_list = list(end_hg38_list)
            length_list = list(length_list)

    else:
        eccid_list = []
        chr_list = []
        start_hg38_list = []
        end_hg38_list = []
        length_list = []

        for i in range(len(data_frame)):
            generate_single_false(i, data_frame, eccid_list, chr_list,
                                  start_hg38_list, end_hg38_list, length_list, around_len, species, seed)

    false_data_frame = pd.DataFrame({
        'eccid': eccid_list,
        'chr_hg38': chr_list,
        'start_hg38': start_hg38_list,
        'end_hg38': end_hg38_list,
        'Length': length_list
    })

    return false_data_frame


def filter_list_out_of_range(list1, list2, tolerance, distance=1):
    result = []
    tolerance_ = tolerance * distance
    for num in list1:
        if all(abs(num - ref) > tolerance_ for ref in list2):
            result.append(num)
    return result


def merge_chr_data(data_path):
    file_names = os.listdir(data_path)
    out_df = None
    for file in file_names:
        file_path = os.path.join(data_path, file)
        file_data = pd.read_csv(file_path)
        if out_df is None:
            out_df = file_data
        else:
            out_df = pd.concat([out_df, file_data], axis=0, ignore_index=True)
    return out_df


def single_chr_false_data(data_frame, index, around_len, start_end='start',
                           species='human', seed=42, multi_process=False, distance=1):
    chr_list = []
    if start_end == 'start':
        seed_ = seed + index
    else:
        seed_ = seed + index + 24
    rng = np.random.default_rng(seed=seed_)
    chr_ = data_frame['chr_hg38'].iloc[0]

    if start_end == 'start':
        sub_df_point = list(data_frame['start_hg38'])
    else:
        sub_df_point = list(data_frame['end_hg38'])
    sub_df_length = len(data_frame)

    if species == 'human':
        species_chr_length = chromosome_length[chr_]
    elif species == 'mouse':
        species_chr_length = chromosome_length_mouse[chr_]
    elif species == 'gallus':
        species_chr_length = chromosome_length_gallus[chr_]
    else:
        raise KeyError(f'No such species')

    random_down = around_len
    random_upper = species_chr_length - around_len
    random_point = rng.integers(low=random_down, high=random_upper, size=sub_df_length)
    random_point = filter_list_out_of_range(random_point, sub_df_point, tolerance=around_len, distance=distance)

    while len(random_point) < sub_df_length * 0.99:
        append_size = sub_df_length - len(random_point)
        append_point = rng.integers(low=random_down, high=random_upper, size=append_size)
        append_point = filter_list_out_of_range(append_point, sub_df_point, tolerance=around_len, distance=distance)
        random_point.extend(append_point)

    chr_list.extend([chr_] * len(random_point))

    if multi_process:
        df_save = pd.DataFrame({
            'chr_hg38': chr_list,
            f'{start_end}_hg38': random_point,

        })
        save_path = f'./temp_data_{seed}/temp_{chr_}_{start_end}.csv'
        df_save.to_csv(save_path, index=False, header=True)
    else:
        return chr_list, random_point


def generate_false_data_(data_frame, around_len=3000, multi_process=False, species='human', seed=42):
    around_len = int(around_len // 2)
    data_frame = data_frame.sort_values(by=['chr_hg38'], ascending=True)
    unique_num = len(np.unique(data_frame['chr_hg38']))
    split_dfs = [group for _, group in data_frame.groupby('chr_hg38')]

    if multi_process:
        start_ = 'start'
        end_ = 'end'
        dir_name = f'./temp_data_{seed}/'
        os.makedirs(dir_name, exist_ok=True)
        with Pool(processes=24) as p:
            p.starmap(single_chr_false_data,
                      [(split_dfs[i], i, around_len, start_, species, seed, multi_process) for i in range(unique_num)])
        df_start = merge_chr_data(dir_name)
        shutil.rmtree(dir_name)

        os.makedirs(dir_name, exist_ok=True)
        with Pool(processes=24) as p:
            p.starmap(single_chr_false_data,
                      [(split_dfs[i], i, around_len, end_, species, seed, multi_process) for i in range(unique_num)])
        df_end = merge_chr_data(dir_name)
        shutil.rmtree(dir_name)

    else:
        start_hg38_list = []
        end_hg38_list = []
        chr_list_start = []
        for i in range(unique_num):
            single_chr_list, random_points = single_chr_false_data(split_dfs[i], i, around_len=around_len,
                                                                   start_end='start', species=species, seed=seed)
            chr_list_start.extend(single_chr_list)
            start_hg38_list.extend(random_points)

        chr_list_end = []
        for i in range(unique_num):
            single_chr_list, random_points = single_chr_false_data(split_dfs[i], i, around_len=around_len,
                                                                    start_end='end', species=species, seed=seed)
            chr_list_end.extend(single_chr_list)
            end_hg38_list.extend(random_points)
        df_start = pd.DataFrame({
            'chr_hg38': chr_list_start,
            'start_hg38': start_hg38_list,

        })
        df_end = pd.DataFrame({
            'chr_hg38': chr_list_end,
            'end_hg38': end_hg38_list,
        })

    df_start['group'] = df_start.groupby('chr_hg38').cumcount()
    df_end['group'] = df_end.groupby('chr_hg38').cumcount()
    out_df = pd.merge(df_start, df_end, on=['chr_hg38', 'group'], how='inner').drop('group', axis=1)
    eccid_list = ['eccid_false_' + str(i) for i in range(len(out_df))]
    length_list = [100] * len(out_df)
    out_df['eccid'] = eccid_list
    out_df['Length'] = length_list
    out_df = out_df[['eccid', 'chr_hg38', 'start_hg38', 'end_hg38', 'Length']]
    return out_df


def generate_pure_random_data(counts=500000, species='human', seed=42):
    rng = np.random.default_rng(seed=seed)
    if species == 'human':
        chr_dict = chromosome_length
    elif species == 'mouse':
        chr_dict = chromosome_length_mouse
    elif species == 'gallus':
        chr_dict = chromosome_length_gallus
    else:
        raise KeyError(f'No such species')

    total_length = sum(chr_dict.values())
    counts_dict = {}
    for chrom, length in chr_dict.items():
        proportion = length / total_length
        chr_count = int(proportion * counts)
        counts_dict[chrom] = chr_count

    chr_list = []
    start_list = []
    end_list = []
    for chrom, count in counts_dict.items():
        species_chr_length = chr_dict[chrom]

        random_down = 0
        random_upper = species_chr_length
        random_start = rng.integers(low=random_down, high=random_upper, size=count)
        random_end = rng.integers(low=random_down, high=random_upper, size=count)
        chrs = [chrom] * count
        chr_list.extend(chrs)
        start_list.extend(random_start)
        end_list.extend(random_end)

    all_df_length = len(start_list)
    eccid_list = ['eccid_random_' + str(i) for i in range(all_df_length)]
    df_length = [100] * all_df_length

    out_df = pd.DataFrame({
        'eccid': eccid_list,
        'chr_hg38': chr_list,
        'start_hg38': start_list,
        'end_hg38': end_list,
        'Length': df_length
    })
    return out_df


def shuffle_positive_data(pos_path):
    if pos_path.endswith('.csv'):
        eccDNA_raw_data = pd.read_csv(pos_path)
    else:
        eccDNA_raw_data = pd.read_table(pos_path, sep=' ')

    pos_data_0 = eccDNA_raw_data[['eccid', 'chr_hg38',  'Length', 'ac_number']]
    pos_data_1 = eccDNA_raw_data[['start_hg38', 'around_start']]
    pos_data_2 = eccDNA_raw_data[['end_hg38', 'around_end']]

    pos_data_1 = pos_data_1.sample(frac=1).reset_index(drop=True)
    pos_data_2 = pos_data_2.sample(frac=1).reset_index(drop=True)

    eccDNA_raw_data = pd.concat([pos_data_0, pos_data_1, pos_data_2], axis=1)
    eccDNA_raw_data = eccDNA_raw_data[['eccid', 'chr_hg38', 'start_hg38', 'end_hg38', 'Length', 'ac_number',
                             'around_start', 'around_end']]
    return eccDNA_raw_data





