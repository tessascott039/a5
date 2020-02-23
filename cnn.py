#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    ### YOUR CODE HERE for part 1g

    def __init__(self, char_embed_size, word_embed_size, window_size=5):
        super().__init__()
        self.conv1d = nn.Conv1d(char_embed_size, word_embed_size, kernel_size=window_size, bias=True, padding=1)

    def forward(self, x_emb: torch.Tensor):
        x_conv = self.conv1d(x_emb)
        x_conv_out = torch.max(F.relu(x_conv), dim=2)[0]
        return x_conv_out

    ### END YOUR CODE

