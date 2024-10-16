import numpy as np
import torch
from collections import defaultdict
from typing import List, Tuple


class NAryBooleanExpLang(object):

    def __init__(self, n: int, p = 0.5):
        self.n = n
        self.prob = p
        self.operands = ['0', '1']
        symbols = ['~', '+', '>', '*', '^']
        # The boolean operators filtered out depending on the `n`-ary input
        self.operators = symbols[:n]

    def get_random_symbol(self)-> Tuple[str, int]:
        # Returns an operator / operand, along with a modified index of where it was found
        # Choices is a list : [0, 1, 2 .. n]
        choices = [i for i in range(self.n + 1)]
        # Assign high probability to 0 : which chooses operands, instead of new operators ~/+ ..
        prob_split = [1 - self.prob] + [self.prob/self.n for i in range(self.n)]
        # Choose an index which will be used to select either an operator / some operands
        expn_choice_index = np.random.choice(choices, p = prob_split)
        # If index chosen is 0, give preference to operands
        if expn_choice_index == 0:
            # Choose one of the operands and return
            return np.random.choice(self.operands), expn_choice_index
        # Since, index = 0 is reserved for values, for getting correct operator and index
        # decrement the index for retrieving operators
        return self.operators[expn_choice_index - 1], expn_choice_index

    def generate_string(self, max_length: int) -> str:
        incomplete_operand_count, generated_expression = 1, ''
        # Try to create a maximal expression (which is unlikely)
        for _ in range(max_length):
            # Generate a new symbol
            symbol, symbol_index = self.get_random_symbol()
            # Add it to the overall expression
            generated_expression += symbol
            # If new symbol is an operand, symbol_index = 0, hence count will be decremented by 1
            # If new symbol is an operator, depending on the arity of the operator, the count will get increased
            # Eg: for `+`, arity is 2, hence count will be increased by 1, since by default, count starts from 1
            # It will require 2 operands to bring it back to 0. This holds true for all operators, coming at any position
            incomplete_operand_count = incomplete_operand_count - 1 + symbol_index
            # If the incomplete operands are all fullfilled, i.e. value drops to 0, return the generated expression
            if incomplete_operand_count == 0:
                return generated_expression
        # If we couldn't create a valid expressoin, return an empty expression, instead of an incorrect one
        return ''

    def generate_list(self, num: int, min_size: int, max_size: int) -> List[str]:
        input_samples, num_samples_added = set(), 0
        while num_samples_added < num:
            bool_expr = self.generate_string(max_size)
            # check if minimum size constraints aren't broken, and if the element already exists
            if len(bool_expr) < min_size or bool_expr in input_samples or bool_expr == '':
                continue
            num_samples_added += 1
            input_samples.add(bool_expr)
        return list(input_samples)

    def output_generator(self, seq: str) -> str:
        # 0's for the entire sequence except the last entry
        return ''.join(['0' for i in range(len(seq) - 1)]) + '1'

    def training_set_generator(self, num: int, min_size: int, max_size: int) -> Tuple[List[str], List[str]]:
        input_seq = self.generate_list(num, min_size, max_size)
        # Generate the output for each sample in the input sequence
        output_seq = [self.output_generator(inp_sample) for inp_sample in input_seq]
        return input_seq, output_seq
