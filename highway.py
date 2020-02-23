#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    ### YOUR CODE HERE for part 1f
    #use two linear layers

    def __init__(self, word_embed_size, dropout_val):
        super().__init__()
        self.proj = nn.Linear(word_embed_size, word_embed_size)
        self.gate = nn.Linear(word_embed_size, word_embed_size)
        self.dropout = nn.Dropout(dropout_val)

    def forward(self, input: torch.Tensor):
        x_gate = torch.sigmoid(self.gate(input))
        x_proj = F.relu(self.proj(input))
        x_highway = x_gate * x_proj + (1 - x_gate) * input
        x_word_emb = self.dropout(x_highway)
        return x_word_emb

    ### END YOUR CODE