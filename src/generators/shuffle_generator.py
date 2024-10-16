import sys
import random
import torch
import textwrap

import numpy as np
from collections import defaultdict
from typing import Tuple, List, Dict

sys.setrecursionlimit(5000)


class ShuffleLanguage():
    def __init__(self, num_pairs: int, p: float, q: float)-> None:
        # Number of pairs to consider 
        self.pair_num = num_pairs
        # Total possible character pairs in shuffle 
        all_pairs = ['()', '[]', '{}', '<>', '+-', 'ab', 'xo']
        # Take a subset of the number of possible pairs, as given by the input
        self.pairs = all_pairs[:num_pairs]
            # The vocabulary is created from the restricted pairs
        self.vocabulary = ''.join([i for i in all_pairs])
        self.n_letters = len(self.vocabulary)

        # Open parameters are always at the start of every string in all_pairs
        self.openpar= [elt[0] for elt in self.pairs]
        self.closepar = [elt[1] for elt in self.pairs]

        # Probabilities according to which the shuffling will take place
        self.p = p
        self.q = q
    
    def return_vocab(self):
        return self.vocabulary

    def generate(self, current_size: int, max_size: int, max_depth: int)-> str:
        # Return, as the size has already been crossed, and so, nothing else should be added
        if current_size >= max_size: 
            return ''
        
        # Get a number between 0 and 1
        prob = random.random()
        # Grammar: S -> (_i S )_i with prob < p | SS with prob (p,q) | empty with prob 1-(p+q)
        if prob < self.p:
            # randomly pick one of the pairs.
            chosen_pair = np.random.choice(self.pairs)
            # Recursively add another substring in the middle of the start and end of the bracket
            # Increment the current_size by 2, as 2 characters have already been added
            sample = chosen_pair[0] + self.generate (current_size + 2, max_size, max_depth) + chosen_pair [1]
            # Sanity check, in case the length of the generated string crosses the maximum size, nothing is returned
            if len(sample) <= max_size:
                # In case, no depth constraint existed, or depth constraints were followed, return
                if max_depth == -1 or len(sample)==0 or self.get_depth_at_each_point_in_seq(sample).sum(axis=1).max() <= max_depth:
                    return sample
        elif prob < self.p + self.q:
            # Recursively add 2 substrings independently of each other and concatenate them
            # No addition of characters in this loop, hence no increment in the current size
            sample = self.generate(current_size, max_size, max_depth) + self.generate(current_size, max_size, max_depth)
            # Sanity check, in case the length of the generated string crosses the maximum size, nothing is returned
            if len (sample) <= max_size:
                # In case, no depth constraint existed, or depth constraints were followed, return
                if max_depth == -1 or len(sample)==0 or self.get_depth_at_each_point_in_seq(sample).sum(axis=1).max() <= max_depth:
                    return sample
        return ''

    def generate_list(self, to_generate_num: int, min_size: int, max_size: int,
                      min_depth: int=0, max_depth: int=-1)-> Tuple[List[str], Dict[int, List[str]]]:
        curr_generated_num = 0
        generated_seq_list = []
        # Group sequences according to their lengths
        size_info = defaultdict(list)

        # Keep trying to generate sequences, till the required number of sequences are satisfied
        while curr_generated_num < to_generate_num:
            # Generate samples as per the given maximum depth
            sample = self.generate(0, max_size, max_depth)
            # If sample is not a duplicate, and is bigger than the smallest allowed size
            if sample not in generated_seq_list and len(sample) >= min_size:
                # If either maximum depth wasn't specified, or the minimum depth is satisified
                if max_depth==-1 or self.get_depth_at_each_point_in_seq(sample).sum(1).max() >= min_depth:
                    # Increment the number of accepted generations, add the generated list to the final sequence list
                    curr_generated_num += 1
                    generated_seq_list.append(sample)
                    # Keep track of the number of samples generated, according to their length
                    size_info[len(sample)].append(sample)
                    print ('{}/{} samples generated.'.format(curr_generated_num, to_generate_num), end = '\r', flush = True)
        print()
        return generated_seq_list, size_info

    def output_generator(self, seq:str)-> str:
        depths = self.get_depth_at_each_point_in_seq(seq)
        # Output is pair_num * input_len, hence at each point in the input
        # Check all the pairs, if the pair is closed, Add a 0 there, else add a 1
        output_seq = ''.join(['0' if depths[i][j] == 0 else '1' for i in range(len(seq)) for j in range(self.pair_num)])
        return output_seq

    def get_depth_at_each_point_in_seq(self, seq: str)-> np.ndarray:
        dyck_counter = np.zeros(self.pair_num)
        max_depth = np.zeros((len(seq), self.pair_num))

        for index, elt in enumerate(seq):
            if elt in self.openpar:
                # If a open bracket found, increment the index of the pair
                dyck_counter[self.openpar.index(elt)] += 1
            else:
                # If a closed bracket was found, decrease the index of the pair
                dyck_counter[self.closepar.index(elt)] -= 1
            # Save the status of the indices at every single character in the sequence
            max_depth[index] = dyck_counter
        return max_depth

    def training_set_generator(self, to_generate_num: int, min_size: int, max_size: int,
                               min_depth: int=0, max_depth: int=-1) -> Tuple[List[str], List[str], Dict[int, List[str]]]:
        # Generate the input sequence list
        input_arr, input_size_arr = self.generate_list(to_generate_num, min_size, max_size, min_depth, max_depth)
        # Per input, get the corresponding output
        output_arr = [self.output_generator(seq) for seq in input_arr]
        # Return everything
        return input_arr, output_arr, input_size_arr
