import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict


class NonStarFreeLanguage(object):

    def __init__(self, n: int) -> None:
        # Total number of letters, from which languages will be created
        letters = ['a','b','c','d','e','f','g','h']
        # Select a small subset of letters as per the input to the class
        self.sigma = letters[:n]
        # Create mapping to indices
        self.char2id = {ch:i for i,ch in enumerate(self.sigma)}
        # Total vocabulary size
        self.n_letters = n

    @abstractmethod
    def belongToLang(self, seq: str) -> bool:
        pass

    @abstractmethod
    def generate_string(self, min_length: int, max_length: int) -> str:
        pass

    def generate_list(self, to_generate_num: int, min_length: int, max_length: int) -> List[str]:
        final_list = []
        while len(final_list) < to_generate_num:
            string = self.generate_string(min_length, max_length)
            # If generated string within the allowed bounds, add it to the list
            if min_length <= len(string) <= max_length:
                final_list.append(string)
                print("Generated {}/{} samples".format(len(final_list), to_generate_num), end = '\r', flush = True)
        print()
        return final_list

    def output_generator(self, seq: str) -> str:
        return ''.join(['1' if self.belongToLang(seq[:i]) else '0' for i in range(1, len(seq) + 1)])

    def training_set_generator(self, to_generate_num: int, min_size: int, max_size: int) -> Tuple[List[str], List[str]]:
        # Generate `to_generate_num` number of input and output samples
        input_arr = self.generate_list(to_generate_num, min_size, max_size)
        output_arr = [self.output_generator(seq) for seq in input_arr]
        return input_arr, output_arr


class ABABStarLanguage(NonStarFreeLanguage):

    def __init__(self, n : int = 2) -> None:
        super(ABABStarLanguage, self).__init__(n)
    
    def belongToLang(self, seq: str) -> bool:
        # Since AB - AB, the sublength is 2X the vocab
        sublen = self.n_letters * 2

        # We should be able to divide entire sequence into subsequences of 2Xn_letters each (eg: abab)
        if len(seq) % sublen != 0:
            return False

        # Check if each subsequence is in fact ABAB or SigmaSigma
        for i in range(0, len(seq), sublen):
            subseq = seq[i:i+sublen]
            if subseq != ''.join(self.sigma + self.sigma):
                return False

        return True

    def generate_string(self, min_length: int, max_length: int) -> str:
        sublen = self.n_letters * 2
        num_ababs = (min_length + np.random.randint(max_length - min_length + 1))//sublen
        # Join strings of the form `AB AB` random numbexr of times (with min and max length constraints)
        string = ''.join([''.join(self.sigma+self.sigma) for _ in range(num_ababs)])
        return string


class AAStarLanguage(NonStarFreeLanguage):

    def __init__(self, n: int) -> None:
        super(AAStarLanguage, self).__init__(n = 1)
        self.n = n

    def belongToLang(self, seq: str) -> bool:
        # Similar logic as `ABABStarLanguage's belongToLang` except that this deals with half the len
        req_subseq = ''.join([self.sigma[0] for _ in range(self.n)])
        sublen = len(req_subseq)

        if len(seq) % sublen != 0:
            return False

        for i in range(0, len(seq), sublen):
            subseq = seq[i:i+sublen]
            if subseq != req_subseq:
                return False
        
        return True

    def generate_string(self, min_length: int, max_length: int) -> str:
        req_subseq = ''.join([self.sigma[0] for _ in range(self.n)])
        sublen = len(req_subseq)
        num_aas = (min_length + np.random.randint(max_length - min_length + 1))//sublen
        # Join strings of the form `AA` random numbexr of times (with min and max length constraints)
        string = ''.join([''.join(req_subseq) for _ in range(num_aas)])
        return string


class AnStarA2Language(NonStarFreeLanguage):

    def __init__(self, n: int) -> None:
        super(AnStarA2Language, self).__init__(n = 1)
        self.n = n
        self.lang = AAStarLanguage(n)

    def generate_string(self, min_length: int, max_length: int) -> str:
        string =  self.lang.generate_string(min_length, max_length) + 'aa'
        return string

    def belongToLang(self, seq: str) -> bool:
        if len(seq) < 2:
            return False
        if seq[-2:] != 'aa':
            return False
        return self.lang.belongToLang(seq[:-2])