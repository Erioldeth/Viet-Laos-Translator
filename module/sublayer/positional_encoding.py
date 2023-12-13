import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):

	def __init__(self, d_model, dropout, max_len):
		super().__init__()
		self.dropout = nn.Dropout(dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float)[:, None]
		div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe[None]
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:, :x.size(1)].requires_grad_(False)
		return self.dropout(x)
