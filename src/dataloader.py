import torch
from src.generators import tomita_generator
from src.generators import starfree_generator
from src.generators import nonstarfree_generator
from src.generators.starfree_generator import AB_D_BC, ZOT_Z_T
from src.generators.parity_generator import ParityLanguage
from src.generators.shuffle_generator import ShuffleLanguage
from src.generators.counter_generator import CounterLanguage
from src.generators.boolean_generator import NAryBooleanExpLang
from src.generators.dyck_generator import DyckLanguage
from src.sentence_processing import sents_to_idx
from typing import List, Tuple, Dict


class DyckCorpus(object):
    def __init__(self, p_val, q_val, num_par, lower_window, upper_window, size, min_depth=0, max_depth=-1, debug=False):

        if debug:
            size =100

        self.Lang = DyckLanguage(num_par, p_val, q_val)
        self.source, self.target, st = self.generate_data(size, lower_window, upper_window, min_depth, max_depth)
        lx = [len(st[z]) for z in list(st.keys())]
        self.st =st

    def generate_data(self, size, lower_window, upper_window, min_depth, max_depth):
        inputs, outputs, st = self.Lang.training_set_generator(size, lower_window, upper_window, min_depth, max_depth)
        return inputs, outputs, st


class TomitaCorpus(object):
    def __init__(self, n, lower_window, upper_window, size, unique, leak=False, debug=False):
        assert n > 0 and n <= 7
        L = (lower_window + upper_window) // 2
        p = L / (2 * (1 + L))
        self.unique = unique
        self.leak = leak
        self.Lang = getattr(tomita_generator, 'Tomita{}Language'.format(n))(p, p)
        self.source, self.target = self.generate_data(size, lower_window, upper_window)

    def generate_data(self, size, lower_window, upper_window):
        inputs, outputs = self.Lang.training_set_generator(size, lower_window, upper_window, self.leak)

        if self.unique:
            inputs, outputs = zip(*set(zip(inputs, outputs)))
            inputs = list(inputs)
            outputs = list(outputs)

        return inputs, outputs


class ParityCorpus(object):
    def __init__(self, lower_window: int, upper_window: int, size: int) -> None:
        # The probabilities are slightly lower than 0.5
        # eg: 4.9 for adding 0, 4.9 for adding 1, and 0.2 for ending the string
        L = (lower_window + upper_window) // 2
        p = L / (2 * (1 + L))
        self.Lang = ParityLanguage(p, p)
        # Generate input and output `size` samples
        self.source, self.target = self.generate_data(size, lower_window, upper_window)

    def generate_data(self, size: int, lower_window: int, upper_window: int):
        inputs, outputs = self.Lang.training_set_generator(size, lower_window, upper_window)
        return inputs, outputs


class StarFreeCorpus(object):

    def __init__(self, lang, num_par: int, lower_window: int, upper_window: int, size: int, unique: bool = False):
        # num_par -> refers to the depth, eg: in case of D languages, D2, D4, etc. 
        self.Lang = getattr(starfree_generator, lang+'Language')(num_par)
        # Remove duplicate entries from the dataset
        self.unique = unique
        self.source, self.target = self.generate_data(size, lower_window, upper_window)

    def generate_data(self, size: int, lower_window: int, upper_window: int)-> Tuple[List[str], List[str]]:
        inputs, outputs = self.Lang.training_set_generator(size, lower_window, upper_window)
        if self.unique:
            # Remove duplicate entries from the datasets
            inputs, outputs = zip(*set(zip(inputs, outputs)))
            inputs, outputs = list(inputs), list(outputs)
        return inputs, outputs


class StarFreePostLanguageCorpus(object):
    def __init__(self, mandatory: str, pre_choices: str, post_choices: str, lower_window: int, upper_window: int, size: int):        
        if mandatory == 'd':
            self.lang = AB_D_BC(pre_choices, post_choices, mandatory)
        elif mandatory == '0':
            self.lang = ZOT_Z_T(pre_choices, post_choices, mandatory)
        self.source, self.target = self.generate_data(size, lower_window, upper_window)

    def generate_data(self, size: int, lower_window: int, upper_window: int)-> Tuple[List[str], List[str]]:
        inputs, outputs = self.lang.training_set_generator(size, lower_window, upper_window)
        # Remove duplicate entries from the datasets
        inputs, outputs = zip(*set(zip(inputs, outputs)))
        return list(inputs), list(outputs)


class NonStarFreeCorpus(object):
    def __init__(self, lang, num_par: int, lower_window: int, upper_window: int, size: int, unique: bool = False):
        # num_par : number of repititions in case of AAStar, AA: 2, AAAA: 4 | for ABAB, its number of characters
        self.Lang = getattr(nonstarfree_generator, lang + 'Language')(num_par)
        # To remove duplicate entries from the dataset
        self.unique = unique
        self.source, self.target = self.generate_data(size, lower_window, upper_window)

    def generate_data(self, size: int, lower_window: int, upper_window: int) -> Tuple[List[str], List[str]]:
        inputs, outputs = self.Lang.training_set_generator(size, lower_window, upper_window)
        if self.unique:
            # Remove duplicate entries from the dataset
            inputs, outputs = zip(*set(zip(inputs, outputs)))
            inputs, outputs = list(inputs), list(outputs)
        return inputs, outputs


class ShuffleCorpus(object):
    def __init__(self, p_val: float, q_val: float, num_par: int, lower_window: int,
                 upper_window: int, size: int, min_depth: int=0, max_depth: int=-1) -> None:

        # Initialise the Shuffle language class to generate samples (num_par : number of bracket pairs)
        self.Lang = ShuffleLanguage(num_par, p_val, q_val)
        # Generate the inputs and outputs from the Shuffle language family using the given parameters
        self.source, self.target, st = self.generate_data(size, lower_window, upper_window, min_depth, max_depth)
        # Grouped inputs by Size
        self.st = st

    def generate_data(self, size: int, lower_window: int, upper_window: int,
                      min_depth: int, max_depth: int) -> Tuple[List[str], List[str], Dict[int, List[str]]]:
        inputs, outputs, st = self.Lang.training_set_generator(size, lower_window, upper_window, min_depth, max_depth)
        return inputs, outputs, st


class CounterCorpus(object):
    def __init__(self, num_par: int, lower_window: int, upper_window: int, size: int) -> None:
        # Initialise counter languages (num_par: number of characters, eg: 2 : a,b ; 3 : a,b,c)
        self.Lang = CounterLanguage(num_par)
        self.source, self.target = self.generate_data(size, lower_window, upper_window)

    def generate_data(self, size: int, lower_window: int, upper_window: int)-> Tuple[List[str], List[str]]:
        # Generate the given number of samples
        inputs, outputs = self.Lang.generate_sample(size, lower_window, upper_window)
        # Remove duplicates after generation
        inputs, outputs = zip(*set(zip(inputs, outputs)))
        return list(inputs), list(outputs)


class BooleanExprCorpus(object):
    def __init__(self, p: float, n: int, lower_window: int, upper_window: int, size: int):
        self.Lang = NAryBooleanExpLang(n = n, p = p)
        self.source, self.target = self.generate_data(size, lower_window, upper_window)

    def generate_data(self, size: int, lower_window: int, upper_window: int):
        inputs, outputs = self.Lang.training_set_generator(size, lower_window, upper_window)
        return inputs, outputs


def dump_data_to_file(data_dir, file_name, corpus):
    """
    Dump corpus to train and target files.
    Args:
        data_dir: name of the directory to save the corpus
        file_name: name of the file
        corpus: train or validation corpus generated
    """
    path = str(data_dir) + "/" + file_name
    with open(path, 'w') as file:
        file.write('\n'.join(corpus))
