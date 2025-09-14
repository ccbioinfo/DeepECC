from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from Bio import SeqIO

normal_chromosome = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
                     'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                     'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']
chromosome_length = {
    'chr1':248956422, 'chr2':242193529, 'chr3':198295559, 'chr4':190214555, 'chr5':181538259,
    'chr6':170805979, 'chr7':159345973, 'chr8':145138636, 'chr9':138394717, 'chr10':133797422,
    'chr11':135086622, 'chr12':133275309, 'chr13':114364328, 'chr14':107043718, 'chr15':101991189,
    'chr16':90338345, 'chr17':83257441, 'chr18':80373285, 'chr19':58617616, 'chr20':64444167, 'chr21':46709983,
    'chr22':50818468, 'chrX':156040895, 'chrY':57227415
}


def load_single_data(data_path, pos_neg='pos', shuffled=False, filtering=False, exchange=False, joint=False):
    if data_path.endswith('.h5'):
        data = pd.read_hdf(data_path, key='data')
    elif data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .h5 or .csv file.")

    if pos_neg == 'pos':
        data['label'] = int(1)
    elif pos_neg == 'neg':
        data['label'] = int(0)
        if exchange:
            data['around_end'] = data['around_start']
    else:
        raise ValueError

    if filtering:
        data = data[data['Length'] > 1e5]

    if shuffled:
        data = data.sample(frac=1).reset_index(drop=True)

    if joint:
        data['around_start'] =  data['around_start'] + data['around_end']
        data['around_end'] = data['around_start']

    return data


def load_data(pos_path, neg_path, shuffled=False, filtering=False, exchange=False, joint=False):

    if pos_path.endswith('.h5'):
        pos_data = pd.read_hdf(pos_path, key='data')
    elif pos_path.endswith('.csv'):
        pos_data = pd.read_csv(pos_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .h5 or .csv file.")

    if neg_path.endswith('.h5'):
        neg_data = pd.read_hdf(neg_path, key='data')
    elif neg_path.endswith('.csv'):
        neg_data = pd.read_csv(neg_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .h5 or .csv file.")

    if exchange:
        neg_data['around_end'] = neg_data['around_start']

    pos_data['label'] = int(1)
    neg_data['label'] = int(0)

    combined_data = pd.concat([pos_data, neg_data], axis=0)

    if filtering:
        combined_data = combined_data[combined_data['Length'] > 1e5]

    if shuffled:
        combined_data_shuffled = combined_data.sample(frac=1).reset_index(drop=True)
    else:
        combined_data_shuffled = combined_data.copy()

    if joint:
        combined_data_shuffled['around_start'] = combined_data_shuffled['around_start'] + combined_data_shuffled['around_end']
        combined_data_shuffled['around_end'] = combined_data_shuffled['around_start']

    return combined_data_shuffled


def base_to_onehot(data_frame, start_end='start'):
    encodings = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.uint8)

    base_dict = np.array([255] * 256, dtype=np.uint8)
    for i, base in enumerate("ACGT"):
        base_dict[ord(base)] = i

    col = 'around_start' if start_end == 'start' else 'around_end'

    sequences = data_frame[col].astype(str).values
    num_samples = len(sequences)
    # max_len = max(map(len, sequences))
    max_len = len(sequences[0])

    idx_array = np.full((num_samples, max_len), 255, dtype=np.uint8)

    for i, seq in enumerate(sequences):
        seq_encoded = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        idx_array[i, :len(seq_encoded)] = base_dict[seq_encoded]

    valid_mask = idx_array != 255
    onehot_array = np.zeros((num_samples, max_len, 4), dtype=np.uint8)
    onehot_array[valid_mask] = encodings[idx_array[valid_mask]]

    return onehot_array.transpose(0, 2, 1)

def simple_to_onehot(seq):
    encodings = {'A': [1, 0, 0, 0],
                 'C': [0, 1, 0, 0],
                 'G': [0, 0, 1, 0],
                 'T': [0, 0, 0, 1]}
    onehot_vector = [encodings.get(base, [0, 0, 0, 0]) for base in seq]
    onehot_vector = np.array(onehot_vector)
    onehot_vector = onehot_vector.swapaxes(0, 1)
    return onehot_vector


def base_to_onehot_joint(start, end, length, chr_sequence):
    encodings = {'A': [1, 0, 0, 0],
                 'C': [0, 1, 0, 0],
                 'G': [0, 0, 1, 0],
                 'T': [0, 0, 0, 1],
                 'a': [1, 0, 0, 0],
                 'c': [0, 1, 0, 0],
                 'g': [0, 0, 1, 0],
                 't': [0, 0, 0, 1]}
    start_around_seq = chr_sequence.seq[(start - length) : (start + length)]
    end_around_seq = chr_sequence.seq[(end - length) : (end + length)]
    joint_seq = start_around_seq + end_around_seq
    print(joint_seq)
    onehot_vector = [encodings.get(base, [0, 0, 0, 0]) for base in joint_seq]
    onehot_vector = np.array(onehot_vector)
    onehot_vector = onehot_vector.T
    return onehot_vector


class EccDNADataset(Dataset):
    def __init__(self, data_frame, start_end='start'):
        self.data_frame = data_frame

        self.vector = base_to_onehot(self.data_frame, start_end)
        self.label = data_frame['label'].to_numpy()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.vector[idx], self.label[idx]


class EccDNADataset_joint(Dataset):
    def __init__(self, data_frame, joint_length, genome_file, chromosome):

        self.data_frame = data_frame
        self.joint_length = int(joint_length / 4)
        self.fasta_file = SeqIO.to_dict(SeqIO.parse(genome_file, "fasta"))
        self.sequence = self.fasta_file.get(chromosome)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return base_to_onehot_joint(self.data_frame.iloc[idx, 1],
                                    self.data_frame.iloc[idx, 2],
                                    self.joint_length,
                                    self.sequence)


class GenomeDataset(Dataset):
    def __init__(self, genome_file, chr='chr1', length=100):
        self.chr = chr
        self.length = length
        fasta_sequencesq = SeqIO.to_dict(SeqIO.parse(genome_file, "fasta"))
        ac = chr[3:]
        self.chr_sequence = fasta_sequencesq.get(ac)

    def __len__(self):
        all_len = chromosome_length[self.chr]
        len_ = all_len - self.length
        return len_

    def __getitem__(self, idx):
        seq_ = self.chr_sequence.seq[idx : idx+self.length]
        seq_ = str(seq_).upper()
        return simple_to_onehot(seq_)


####
####
def create_onehot_lookup():
    table = np.zeros((128, 4), dtype=np.float32)
    table[ord('A')] = [1, 0, 0, 0]
    table[ord('C')] = [0, 1, 0, 0]
    table[ord('G')] = [0, 0, 1, 0]
    table[ord('T')] = [0, 0, 0, 1]
    table[ord('a')] = [1, 0, 0, 0]
    table[ord('c')] = [0, 1, 0, 0]
    table[ord('g')] = [0, 0, 1, 0]
    table[ord('t')] = [0, 0, 0, 1]
    return table

ONEHOT_LOOKUP = create_onehot_lookup()


def fast_base_to_onehot_joint(start, end, length, chr_seq_str, chr_len, fixed_len=True):
    s_start = max(0, start - length)
    s_end = min(chr_len, start + length)
    e_start = max(0, end - length)
    e_end = min(chr_len, end + length)

    joint_seq = chr_seq_str[s_start:s_end] + chr_seq_str[e_start:e_end]

    if fixed_len and len(joint_seq) < 4 * length:
        joint_seq = joint_seq.ljust(4 * length, 'N')

    ascii_codes = np.frombuffer(joint_seq.encode('ascii'), dtype=np.uint8)
    onehot = ONEHOT_LOOKUP[ascii_codes]  # shape: (L, 4)
    return onehot.T


class EccDNADataset_joint_(Dataset):
    def __init__(self, data_frame, joint_length, genome_file, chromosome):
        self.data_frame = data_frame
        self.joint_length = int(joint_length / 4)
        self.fasta_seq = str(SeqIO.to_dict(SeqIO.parse(genome_file, "fasta"))[chromosome].seq)
        self.chrom_len = len(self.fasta_seq)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        return fast_base_to_onehot_joint(row[1], row[2], self.joint_length, self.fasta_seq, self.chrom_len)


if __name__ == '__main__':

    genome_file = 'D:\\B_eccDNA\\EccDNA_model\\data\\genome_data\\Homo_sapiens.fasta'
    fasta_file = SeqIO.to_dict(SeqIO.parse(genome_file, "fasta"))
    fasta_seq = fasta_file.get('CM000663.2')

    chrom_len = len(fasta_seq)
    print(chrom_len)

    start = 10000
    end = 20000
    test_ = base_to_onehot_joint(start, end, 10, fasta_seq)
    print(test_)
