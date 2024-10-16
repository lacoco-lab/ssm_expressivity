import torch
import textwrap

import numpy as np
from typing import Tuple, List

letters = ['a','b','c','d','e','f','g','h']


def get_sigma_star(choices, length):
    # Repeat given character 'length number of times
    return ''.join([np.random.choice(choices) for i in range(length)])


class AAStarBBStarLanguage(object):

    def __init__(self, n: int = 5) -> None:
        letters = 'abcdefgh'
        self.possible_chars = letters[:n]
        self.all_chars = self.possible_chars + 'T'
        self.char2id = {ch:i for i,ch in enumerate(self.all_chars)}
        self.n_letters = n + 1

    def generate_string(self, min_length, max_length):
        string = ''
        total_count = max_length - min_length + 1
        for symbol in self.possible_chars:
            # Random count between 0, total_count, in case count > 0
            count = np.random.randint(total_count) + 1 if total_count > 0 else 0
            # Find the actual count of the new symbol to be added
            symb_count = min_length//(self.n_letters-1) + count
            # Add the symbol, the given number of times to the overall string
            string += symb_count*symbol
            total_count = total_count - count
        # Add a T at the end, to signify end of the string
        return string

    def generate_list(self, num: int, min_length: int, max_length: int)-> List[str]:
        input_list = []
        while len(input_list) < num:
            string = self.generate_string(min_length, max_length)
            # If length constraints aren't violated, and string isn't a duplicate
            if (string not in input_list) and (min_length <= len(string) <= max_length):
                # Make sure, all the eligible characters were put into the string
                if self.possible_chars[-1] in string:
                    input_list.append(string)
                    print("Generated {}/{} samples".format(len(input_list), num), end = '\r', flush = True)
        print()
        return input_list

    
    def output_generator(self, sequence: str) -> str:
        # We can have such a mapping, since for a - possible chars : ab, b - bc, c - cd, d - de, e - eT
        # Which is the same as just saying : a - b, b - c, c - d, d - e, e - T (as output can be tokenized, and mapping would be the same)
        # Ignore the last 'T' Symbol, otherwise create the output string by replacing indices with the next symbol
        return ''.join([self.all_chars[self.char2id[symbol] + 1] for symbol in sequence])

    def training_set_generator(self, num: int, min_size: int, max_size: int)-> Tuple[List[str], List[str]]:
        input_arr = self.generate_list(num, min_size, max_size)
        output_arr = [self.output_generator(seq) for seq in input_arr]
        return input_arr, output_arr


class AB_D_BC(object):
    '''Regular expression: mandatory_chars + (optional character repetitions)*'''
    
    def __init__(self, choices_pre, choices_post, mandatory):
        # all_chars = a,b,c,d for {a,b}*d{b,c}* ; 0,1,2 for {0,1,2}*02*
        self.mandatory = mandatory
        self.choices_pre = list(choices_pre)
        self.choices_post = list(choices_post)
        # Language -> {a,b}*d{b,c}* : All possible values of a, b, d are valid
        self.pre_map = '1101'
        # Default value of b, c are the only possibilities in the 2nd half
        self.post_map = '0110'

    def generate_string(self, max_length: int) -> str:
        # Reduce 1 from max length to allow for the mandatory character
        pre_length = np.random.randint(0, max_length - 1)
        pre_string = ''.join([np.random.choice(self.choices_pre) for _ in range(pre_length)])
        
        # Reduce the length of the already generated string to follow overall string constraints
        post_length = np.random.randint(0, max_length - pre_length - 1)
        post_string = ''.join([np.random.choice(self.choices_post) for _ in range(post_length)])
        
        return pre_string + self.mandatory + post_string
    
    def output_generator(self, seq: str) -> str:
        # Find the last occurrence of the mandatory character : works for both our languages
        split_point = seq.rfind(self.mandatory)
        # For the characters before and after the mandatory character, 
        # apply different mapping schemes that account for possible next characters in the sequence
        return ''.join([self.pre_map if index < split_point else self.post_map for index in range(len(seq))])

    def generate_list(self, num: int, min_length: int, max_length: int)-> List[str]:
        input_list = []
        while len(input_list) < num:
            string = self.generate_string(max_length)
            # If length constraints aren't violated, and string isn't a duplicate
            if (string not in input_list) and (min_length <= len(string) <= max_length):
                input_list.append(string)
                print("Generated {}/{} samples".format(len(input_list), num), end = '\r', flush = True)
        print()
        return input_list

    def training_set_generator(self, num: int, min_size: int, max_size: int)-> Tuple[List[str], List[str]]:
        input_arr = self.generate_list(num, min_size, max_size)
        output_arr = [self.output_generator(seq) for seq in input_arr]
        return input_arr, output_arr


class ZOT_Z_T(object):
    '''Regular expression: {012}*02*'''
    
    def __init__(self, choices_pre, choices_post, mandatory):
        # Still useful for generating the string, can be avoided while generating output
        self.mandatory = mandatory
        self.choices_pre = list(choices_pre)
        self.choices_post = list(choices_post)

    def generate_string(self, max_length: int) -> str:
        # Reduce 1 from max length to allow for the mandatory character
        pre_length = np.random.randint(0, max_length - 1)
        pre_string = ''.join([np.random.choice(self.choices_pre) for _ in range(pre_length)])
        
        # Reduce the length of the already generated string to follow overall string constraints
        post_length = np.random.randint(0, max_length - pre_length - 1)
        post_string = ''.join([np.random.choice(self.choices_post) for _ in range(post_length)])
        
        return pre_string + self.mandatory + post_string
    
    def output_generator(self, seq: str) -> str:
        # Find the last occurrence of the mandatory character : works for both our languages
        is_2_in_end_state = False
        output_str = ''
        for s in seq:
            if s == '1':
                is_2_in_end_state = False
            elif s == '0':
                # Switch the state of 2, in case one encounters a 0
                is_2_in_end_state = True
            # If 2 is encountered in the sequence, the status quo is maintained
            
            # If curr char = 0 / 1 OR 2 can't end the sequence then 0,1,2 are possible as the next char
            if s != '2' or is_2_in_end_state is False:
                # Then the input can continue
                output_str += 'c'
            else: 
                # in case both conditions are false, then current char is 2, and its in the end state
                # so 0/1/2/EOS possible
                output_str += 'e'
        return output_str

    def generate_list(self, num: int, min_length: int, max_length: int)-> List[str]:
        input_list = []
        while len(input_list) < num:
            string = self.generate_string(max_length)
            # If length constraints aren't violated, and string isn't a duplicate
            if (string not in input_list) and (min_length <= len(string) <= max_length):
                input_list.append(string)
                print("Generated {}/{} samples".format(len(input_list), num), end = '\r', flush = True)
        print()
        return input_list

    def training_set_generator(self, num: int, min_size: int, max_size: int)-> Tuple[List[str], List[str]]:
        input_arr = self.generate_list(num, min_size, max_size)
        output_arr = [self.output_generator(seq) for seq in input_arr]
        return input_arr, output_arr


class PostOptionLanguage(object):
    '''Regular expression: mandatory_chars + (optional character repetitions)*'''
    
    def __init__(self, choices_pre, choices_post, mandatory):
        # all_chars = a,b,c,d for {a,b}*d{b,c}* ; 0,1,2 for {0,1,2}*02*
        self.mandatory = mandatory
        self.choices_pre = list(choices_pre)
        self.choices_post = list(choices_post)
        if mandatory == '0':
            # Language -> {0,1,2}*02*
            # Default value of '111', as all characters are possible
            self.pre_map = '111'
            # Default value of '001' as only 2 is allowed in the 2nd half
            self.post_map = '001'
        elif mandatory == 'd':
            # Language -> {a,b}*d{b,c}* : All possible values of a, b, d are valid
            self.pre_map = '1101'
            # Default value of b, c are the only possibilities in the 2nd half
            self.post_map = '0110'

    def generate_string(self, max_length: int) -> str:
        # Reduce 1 from max length to allow for the mandatory character
        pre_length = np.random.randint(0, max_length - 1)
        pre_string = ''.join([np.random.choice(self.choices_pre) for _ in range(pre_length)])
        
        # Reduce the length of the already generated string to follow overall string constraints
        post_length = np.random.randint(0, max_length - pre_length - 1)
        post_string = ''.join([np.random.choice(self.choices_post) for _ in range(post_length)])
        
        return pre_string + self.mandatory + post_string
    
    def output_generator(self, seq: str) -> str:
        # Find the last occurrence of the mandatory character : works for both our languages
        split_point = seq.rfind(self.mandatory)
        # For the characters before and after the mandatory character, 
        # apply different mapping schemes that account for possible next characters in the sequence
        return ''.join([self.pre_map if index < split_point else self.post_map for index in range(len(seq))])

    def generate_list(self, num: int, min_length: int, max_length: int)-> List[str]:
        input_list = []
        while len(input_list) < num:
            string = self.generate_string(max_length)
            # If length constraints aren't violated, and string isn't a duplicate
            if (string not in input_list) and (min_length <= len(string) <= max_length):
                input_list.append(string)
                print("Generated {}/{} samples".format(len(input_list), num), end = '\r', flush = True)
        print()
        return input_list

    def training_set_generator(self, num: int, min_size: int, max_size: int)-> Tuple[List[str], List[str]]:
        input_arr = self.generate_list(num, min_size, max_size)
        output_arr = [self.output_generator(seq) for seq in input_arr]
        return input_arr, output_arr


class D_nLanguage(object):

    def __init__(self, n: int) -> None:
        self.n = n
        # n_letters, sigma aren't USED in this class
        self.n_letters = 2
        self.sigma = ['a','b']

        self.total_tries = 0
        self.std_ratio = 0.1
        self.mean_ratio = 0.75

    def generate_d_n(self, n: int, maxlength: int) -> str:
        if n == 0 or maxlength == 0:
            return ''

        d_n = ''
        while len(d_n) < maxlength:
            # Get a random length, smaller than the maximum length of the string
            length_d_n_min_1 = int(maxlength * self.mean_ratio * (self.std_ratio * np.random.randn() + 1))
            # Recursively try and increase the length of the string being added
            d_n_min_1 = self.generate_d_n(n-1, length_d_n_min_1)
            # Add the newly made string to the overall string
            d_n += 'a{}b'.format(d_n_min_1)
        return d_n

    def generate_string(self, maxlength: int) -> str:
        # Generate a random length, which will be used to make a string
        length = int(maxlength * self.mean_ratio * (self.std_ratio * np.random.randn() + 1))
        # Generate a string with the random length and given depth
        return self.generate_d_n(self.n, length)
        
    def find_depth(self, sequence: str)-> int:
        # Find the difference in counts between occurrences of a, and that of b
        return sequence.count('a') - sequence.count('b')

    def get_final_state(self, sequence: str)-> str:
        # Find whether 'a's and 'b's are balanced in the given string
        depth = self.find_depth(sequence)
        # return q_0, if depth is 0, q_n if depth is n, q_i if depth is in between
        return '10' if depth == 0 else '01' if depth == self.n else '11'

    def output_generator(self, seq: str)-> str:
        # For the entire sequence, keep checking the depth at each instance, and accordingly add the output
        return ''.join([self.get_final_state(seq[:i+1]) for i in range(len(seq))])

    def generate_list(self, num: int, min_length: int, max_length: int)-> List[str]:
        input_list = []
        while len(input_list) < num:
            # Generate string, add if it's not a duplicate, and doesn't violate length constraints
            string = self.generate_string(max_length)
            if (string not in input_list) and (min_length <=  len(string) <= max_length):
                # Add to the overall list, keep printing status of the number of strings added
                input_list.append(string)
                print("Generated {}/{} samples".format(len(input_list), num), end = '\r', flush = True)
            else:
                # If we aren't being to add anything to the list, despite multiple tries, change the value of mean_ratio
                self.total_tries += 1
            # If despite multiple tries, dataset generation isn't progressing, change the value of mean ratio
            if self.total_tries > 20000:
                self.total_tries = 0
                self.mean_ratio = self.mean_ratio - 0.02
                # If mean ratio can lead to negative values, return dataset with the current status
                if self.mean_ratio < 0:
                    break

        print()
        return input_list

    def training_set_generator(self, num: int, min_size: int, max_size: int)-> Tuple[List[str], List[str]]:
        input_arr = self.generate_list(num, min_size, max_size)
        output_arr = [self.output_generator(seq) for seq in input_arr]
        return input_arr, output_arr
