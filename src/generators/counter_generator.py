import math
import torch
import random
import numpy as np
from typing import List, Tuple


class CounterLanguage():
    def __init__(self, num_char: int)-> None:
        self.num_char = num_char
        self.chars = ['a', 'b', 'c', 'd']

        # Therefore, if num_char = 3, vocabulary : `abc`
        self.vocabulary = ''.join(self.chars[:num_char])
        self.vocab_size = len(self.vocabulary) 

        # Output vocabulary has an extra T: termination symbol
        self.all_letters = self.vocabulary + 'T' 
        self.n_letters = len(self.all_letters)

    def get_vocab(self):
        return self.vocabulary
    
    def construct_output(self, sample: str) -> str:
        # Character prediction task, In every case, only 1 'b' is less in the output, since
        # we don't know when 'b' is supposed to start, but for every other character, the number
        # remains the same : 'aabbccdd' -> 'a/b a/b b c c d d T' -> 'aabccddT' (since data is going to be 
        # converted to numbers anyways)
        # Hence delete the 2nd character -> 'B' in our case, and add a T at the end
        sample = sample.replace(self.chars[1], '', 1)
        return sample + 'T'

    def generate_sample(self, sample_size: int=1, min_rept: int=1, max_rept: int=50) -> Tuple[List[str], List[str]]:
        # Generates a list of numbers that lie between `min_rept` and `max_rept`
        rand_num_list = np.random.choice(a=list(range(min_rept, max_rept + 1)), size=sample_size)
        # Repeat each input character from the vocabulary n number of times per sample
        # The `n` is retrieved from the randomly generated list is the previous list
        input_arr = [''.join([i for i in self.vocabulary for _ in range(n)]) for n in rand_num_list]
        # Replace 1st character by extra_letter, end string with a 'T'
        output_arr = [self.construct_output(sample) for sample in input_arr]
        return input_arr, output_arr
