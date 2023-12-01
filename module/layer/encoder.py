import math

import torch
import torch.nn as nn

from module.sublayer import *


class Encoder(nn.Module):
	def __init__(self, input_dim, d_model, N, heads, dropout, max_len=200):
		super().__init__()

		self.tok_embed = nn.Embedding(input_dim, d_model)
		self.pos_embed = nn.Embedding(max_len, d_model)

		self.layers = nn.ModuleList([EncoderLayer(d_model, heads, dropout) for _ in range(N)])

		self.dropout = nn.Dropout(dropout)

		# TODO: check use of this
		self._max_seq_length = max_len

	def forward(self, src, src_mask):
		# src = [batch_size, src_len]
		# src_mask = [batch_size, 1, 1, src_len]
		bs, src_len = src.shape

		pos = torch.arange(0, src_len).unsqueeze(0).repeat(bs, 1)
		# pos = [batch_size, src_len]

		x = self.dropout(self.tok_embed(src) * math.sqrt(self.d_model) + self.pos_embed(pos))
		# x = [batch_size, src_len, d_model]

		for layer in self.layers:
			x = layer(x, src_mask)
		# x = [batch_size, src_len, d_model]

		return x


class EncoderLayer(nn.Module):
	def __init__(self, d_model, heads, dropout=0.1):
		super().__init__()

		self.sa_norm = nn.LayerNorm(d_model)
		self.ff_norm = nn.LayerNorm(d_model)

		self.attn = MultiHeadAttention(heads, d_model, dropout)
		self.ff = FeedForward(d_model, dropout)

		self.dropout = nn.Dropout(dropout)

	def forward(self, src, src_mask):
		# src = [batch_size, src_len, d_model]
		# src_mask = [batch_size, 1, 1, src_len]

		src_sa, _ = self.attn(src, src, src, src_mask)

		src = self.sa_norm(src + self.dropout(src_sa))
		# src = [batch_size, src_len, d_model]

		src_ff = self.ff(src)

		src = self.ff_norm(src + self.dropout(src_ff))
		# src = [batch_size, src_len, d_model]

		return src
