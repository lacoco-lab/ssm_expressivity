import sys
import torch
import random
import numpy as np
from typing import Tuple, List, Dict
from collections import defaultdict, Counter


class ParityLanguage():
    def __init__(self, p: float, q: float) -> None:
        self.p = p
        self.q = q
        self.vocabulary = ['0', '1']
        self.pos_symbol = '1'
        self.n_letters = len(self.vocabulary)

    def check_parity(self, w: str) -> bool:
        """ Return true, if input string has even `pos_symbols`"""
        if w == '':
            return True
        counter = Counter(w)
        return counter[self.pos_symbol] % 2 == 0

    def generate_string(self, max_length: int) -> str:
        string = ''
        while len(string) < max_length:
            # Choose 0, 1, 2 based on the given probability split
            # Since p, q are close to 0.5, the probability of ending the string is quite low
            symbol = np.random.choice(3, p = [self.p, self.q, 1-(self.p + self.q)])
            if symbol == 2:
                break
            # Thus either 0 or 1 gets added
            string += str(symbol)
        return string

    def generate_list(self, to_generate_num: int, min_length: int, max_length: int) -> List[str]:
        final_list = []
        while len(final_list) < to_generate_num:
            string = self.generate_string(max_length)
            # If generated string is not a duplicate and it's length is within permissible limits
            if string not in final_list and min_length <= len(string) <= max_length:
                if self.check_parity(string) is False:
                    # Only even length strings are added
                    string += '1'
                # Need to Check length again
                if len(string) <= max_length:
                    final_list.append(string)
        return final_list

    def output_generator(self, seq: str) -> str:
        return ''.join(['1' if self.check_parity(seq[:i]) else '0' for i in range(1, len(seq) + 1)])

    def training_set_generator(self, to_generate_num: int, min_size: int, max_size: int) -> Tuple[List[str], List[str]]:
        input_arr = self.generate_list(to_generate_num, min_size, max_size)
        output_arr = [self.output_generator (seq) for seq in input_arr]
        return input_arr, output_arr
