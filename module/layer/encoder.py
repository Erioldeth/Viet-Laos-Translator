import math

import torch.nn as nn

from module.sublayer import *


class Encoder(nn.Module):
	def __init__(self, input_dim, d_model, N, heads, dropout, max_len, device):
		super().__init__()

		self.tok_embed = nn.Embedding(input_dim, d_model)
		self.pe = PositionalEncoding(d_model, dropout, max_len)

		self.layers = nn.ModuleList([EncoderLayer(d_model, heads, dropout) for _ in range(N)])

		self.norm = nn.LayerNorm(d_model)

		self.device = device

		self.scale = math.sqrt(d_model)

	def forward(self, src, src_mask):
		# src = [batch_size, src_len]
		# src_mask = [batch_size, 1, 1, src_len]

		x = self.pe(self.tok_embed(src) * self.scale)
		# x = [batch_size, src_len, d_model]

		for layer in self.layers:
			x = layer(x, src_mask)

		return self.norm(x)


class EncoderLayer(nn.Module):
	def __init__(self, d_model, heads, dropout):
		super().__init__()
		self.norm = nn.LayerNorm(d_model)

		self.attn = MultiHeadAttention(heads, d_model, dropout)
		self.ff = FeedForward(d_model, dropout)

		self.dropout = nn.Dropout(dropout)

	def forward(self, src, src_mask):
		# src = [batch_size, src_len, d_model]
		# src_mask = [batch_size, 1, 1, src_len]

		x = self.norm(src)
		x, _ = self.attn(x, x, x, src_mask)
		src = src + self.dropout(x)

		x = self.norm(src)
		x = self.ff(x)
		src = src + self.dropout(x)

		return src
