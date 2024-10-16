import os
import torch
import hydra
import datasets
import logging
import pandas as pd
from pathlib import Path
from typing import Union, Dict, List

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from hydra.core.hydra_config import HydraConfig

from src.dataloader import dump_data_to_file
from src.dataloader_utils import (
    create_corpus_parity,
    create_corpus_non_star_free,
    create_corpus_shuffle,
    create_corpus_tomita,
    create_corpus_counter,
    create_corpus_boolean,
    create_corpus_star_free,
    create_corpus_dyck
)


logger = logging.getLogger()
from typing import Union, Dict, List, Tuple


input_map = {'0': 0, '1': 1, '-': 2, 'r': 3, 'i': 4, 'w': 5}
output_map = {'0': [1, 0, 0], '1': [0, 1, 0], '-': [0, 0, 1]}


def tokenize_output_string(output_string, chunk_size) -> List[str]:
    return [output_string[i:i+chunk_size] for i in range(0, len(output_string), chunk_size)]


def get_dataset_reqs(input_strings, output_strings, chunk_size, pad_value):

    all_output_seqs_tokenized = list(map(lambda x: tokenize_output_string(x, chunk_size), output_strings))

    vocab_inp = set([j for i in input_strings for j in i])
    vocab_out = set([j for i in all_output_seqs_tokenized for j in i])

    len_diff = len(vocab_inp) - len(vocab_out)
    if len_diff > 0:
        # this implies that vocab output is smaller than vocab input, hence add pads to the output        
        vocab_out = list(vocab_out) + [pad_value for i in range(len_diff)]
        vocab_inp = list(vocab_inp)
    else:
        # implies that vocab input is smaller than vocab output, hence add pads to the input
        vocab_inp = list(vocab_inp) + [pad_value for i in range(-len_diff)]
        vocab_out = list(vocab_out)

    vocab_inp = pd.get_dummies(vocab_inp)
    vocab_out = pd.get_dummies(vocab_out)

    # Convert character to index
    encoder_inp = vocab_inp.idxmax()
    encoder_out = vocab_out.idxmax()
    
    # Convert index back to character
    decoder_inp = vocab_inp.idxmax(axis=1).to_dict()
    decoder_out = vocab_out.idxmax(axis=1).to_dict()
    return vocab_inp, vocab_out, encoder_inp, encoder_out, decoder_inp, decoder_out



class DatasetClass(Dataset):
    def __init__(self, input_strings, output_strings, vocab_inp, vocab_out, encoder_inp, encoder_out, decoder_inp, decoder_out, chunk_size, is_entropy_loss=True):
        self.input_strings = input_strings
        self.output_strings = output_strings
        # Write the variables to the class for easier access
        self.chunk_size = chunk_size
        self.vocab_inp = vocab_inp
        self.vocab_out = vocab_out
        self.encoder_inp = encoder_inp
        self.encoder_out = encoder_out
        self.decoder_inp = decoder_inp
        self.decoder_out = decoder_out
        self.is_entropy_loss = is_entropy_loss

    def __len__(self) -> int:
        return len(self.input_strings)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Use One-hot encodings if embedding layer isn't present.
        tokenized_output = tokenize_output_string(self.output_strings[idx], self.chunk_size)
        if self.is_entropy_loss is True:
            output = DatasetClass.str_to_tensor_index_input(tokenized_output, self.encoder_out, torch.int)
        else:
            output = DatasetClass.str_to_tensor_one_hot(tokenized_output, self.vocab_out, torch.float)
        return {
            "input": DatasetClass.str_to_tensor_index_input(self.input_strings[idx], self.encoder_inp, torch.int),
            "output": output
        }

    @classmethod
    def str_to_tensor_index_input(cls, input, encoder, dtype=torch.float) -> torch.Tensor:
        # Converts to tensor after converting sequence of chars to indices from the vocabulary
        # + 1, since 0 is reserved for the padding index
        return torch.tensor([encoder[i] + 1 for i in input], dtype=dtype)

    @classmethod
    def str_to_tensor_one_hot(cls, input, vocab, dtype=torch.float) -> torch.Tensor:
        # Converts to tensor after encoding the char to a one hot vector
        return torch.tensor([vocab[i] for i in input], dtype=dtype)


def collate_fn(batch, pad_value=0) -> Union[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    input_list = [i['input'] for i in batch]
    output_list = [i['output'] for i in batch]

    pad_input_list = pad_sequence(input_list, batch_first=True, padding_value=pad_value)
    pad_output_list = pad_sequence(output_list, batch_first=True, padding_value=pad_value)
    return pad_input_list, pad_output_list


def read_file(file_path) -> List[str]:
    with open(file_path) as f:
        data = f.readlines()
        data = [d.replace('\n', '') for d in data]
    return data


def mask_reads_flip_flop(text: str)-> Tuple[List[int], List[int]]:
    # Mask the actual read answers from the input, mask everything except the read answers from the output
    mapped_data = [(input_map['-'], output_map[char]) if idx > 0 and text[idx - 1] == 'r' 
                   else (input_map[char], output_map['-'])
                   for idx, char in enumerate(text)]
    return zip(*mapped_data)


def separate_input_output_flip_flip(batch):
    input_batch, output_batch = zip(*[mask_reads_flip_flop(s) for s in batch['text']])
    return {
        'inputs': torch.tensor(input_batch, dtype=torch.float),
        'outputs': torch.tensor(output_batch, dtype=torch.float)
    }


def get_flip_flop_dataset(batch_size):
    dataset = datasets.load_dataset('synthseq/flipflop')
    dataset.set_transform(separate_input_output_flip_flip)
    # the val_dense is not considered for evaluation, similar to the FLIP FLOP paper
    relevant_keys = ['train', 'val_sparse', 'val']

    # Reduce the training set, to 20X lesser, similar to how it was done for LSTMs in the FLIP FLOP paper 
    # train_rows = int(dataset['train'].num_rows / 20)
    # dataset['train'] = dataset['train'].select(range(train_rows))

    # Reduce to 1000, similar to the FLIP FLOP paper
    dataset['val'] = dataset['val'].select(range(1000))
    # Reduce to 1% of 10^5 for val_sparse, similar to the FLIP FLOP paper
    dataset['val_sparse'] = dataset['val_sparse'].select(range(1000))

    dataloader_dict = {key: DataLoader(dataset[key], batch_size=batch_size) for key in relevant_keys}
    return dataloader_dict


def build_lang_config(dataset_config):
    """
    Build language generator config from hydra config
    Args:
        dataset_config: hydra dataset config

    Returns:
        lang_config: dict of language parameters

    """
    lang_config = {}
    for param, value in dataset_config.items():
        lang_config[param] = value

    return lang_config


def dump_datasets_locally(data_dir, train_dataset, all_val_datasets, data_type):
    # data_type : src / tgt
    dict_key = 'train_' + data_type
    dumped_data_list = [{dict_key: train_dataset}]
    location_file = dict_key + '.txt'
    dump_data_to_file(data_dir, location_file, train_dataset)

    for idx, val_dataset in enumerate(all_val_datasets):
        # Note the validation bin number
        key_str = 'val_' + data_type + '_bin' + str(idx)
        # Write to the file
        dump_data_to_file(data_dir, key_str + ".txt", val_dataset)
        # Append to the list, to be used later for creating dataloaders and datasets
        dumped_data_list.append({key_str: val_dataset})
    return dumped_data_list


def exists_dataset(data_dir):
    files = ['train_src.txt', 'train_tgt.txt']
    for file in files:
        path = str(data_dir) + '/' + file
        if os.path.exists(path) is False:
            return False
    
    print("Data generation aborted. Datasets already exist.")
    return True


# -> Union[Union[Dict, MultipleLenDataset], Union[Dict, SameLenDataset]]:
def create_dataloader(base_folder, batch_size, lang_params=None, is_multiple=False, pad_value=0,
                      max_size=0, num_val_bins=1, generate=False, is_entropy_loss=True):
    dataloader_dict = {}
    dataset_name = base_folder
    base_folder = Path(HydraConfig.get().runtime.cwd) / 'generated_ds' / dataset_name
    chunk_size = lang_params.get('chunk_size', 1)
    curr_lang_fam = lang_params['lang_fam']

    # Generate the dataset, if it doesn't exist already
    if generate and not exists_dataset(base_folder):
        if curr_lang_fam == 'Tomita':
            train_corpus, val_corpus_bins = create_corpus_tomita(lang_params)
        elif curr_lang_fam == 'Parity':
            train_corpus, val_corpus_bins = create_corpus_parity(lang_params)
        elif curr_lang_fam == 'NonStarFree':
            train_corpus, val_corpus_bins = create_corpus_non_star_free(lang_params)
        elif curr_lang_fam == 'Shuffle': 
            train_corpus, val_corpus_bins = create_corpus_shuffle(lang_params)
        elif curr_lang_fam == 'Counter':
            train_corpus, val_corpus_bins = create_corpus_counter(lang_params)
        elif curr_lang_fam == 'Boolean':
            train_corpus, val_corpus_bins = create_corpus_boolean(lang_params)
        elif curr_lang_fam == 'Dyck':
            train_corpus, val_corpus_bins = create_corpus_dyck(lang_params)
        elif curr_lang_fam == 'StarFree':
            train_corpus, val_corpus_bins = create_corpus_star_free(lang_params)

        # Get directory to save generated datasets to
        data_dir = Path(HydraConfig.get().runtime.cwd) / 'generated_ds' / base_folder

        # Join training and validation data together
        problem_data = dump_datasets_locally(data_dir, train_corpus.source, [v.source for v in val_corpus_bins], 'src')
        solution_data = dump_datasets_locally(data_dir, train_corpus.target, [v.target for v in val_corpus_bins], 'tgt')

    # Common pipeline, for reusing data that has been generated already, or for newly generated data
    problem_files = {'train': base_folder / "train_src.txt"}
    solution_files = {'train': base_folder / "train_tgt.txt"}

    # Add the given number of validation files, and create dataloaders out of them as well
    for bin_number in range(num_val_bins):
        bin_number = str(bin_number)
        key_str = 'val_bin' + bin_number
        problem_files[key_str] = base_folder / ('val_src_bin' + bin_number + '.txt')
        solution_files[key_str] = base_folder / ('val_tgt_bin' + bin_number + '.txt')

    # Read in the input & output for the given problem at hand
    problem_data_dict = {category: read_file(f) for category, f in problem_files.items()}
    solution_n_data_dict = {category: read_file(f) for category, f in solution_files.items()}

    for category, problem_data in problem_data_dict.items():
        if max_size > 0:
            # Restrict the length of sequences to be only the given size (For debugging)
            problem_data = [p[:max_size] for p in problem_data]
            solution_data = [p[:max_size] for p in solution_n_data_dict[category]]
        else:
            solution_data = solution_n_data_dict[category]

        if category == 'train':
            # Reuse the vocab, encoder, decoder across all validation datasets to avoid conflicts
            vocab_inp, vocab_out, encoder_inp, encoder_out, decoder_inp, decoder_out = get_dataset_reqs(problem_data, solution_data, chunk_size, pad_value)
        
        dataset = DatasetClass(problem_data, solution_data, vocab_inp, vocab_out, encoder_inp, encoder_out, decoder_inp, decoder_out, chunk_size, is_entropy_loss)
        dataloader_dict[category] = DataLoader(dataset, batch_size=batch_size, collate_fn= lambda x: collate_fn(x, pad_value=pad_value))
    
    return dataloader_dict, dataset
