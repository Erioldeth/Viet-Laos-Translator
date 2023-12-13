import math

import torch.nn as nn

from module.sublayer import *


class Encoder(nn.Module):
	def __init__(self, input_dim, d_model, N, heads, dropout, max_len, device):
		super().__init__()

		self.tok_embed = nn.Embedding(input_dim, d_model)
		# self.pos_embed = nn.Embedding(max_len, d_model)
		self.pe = PositionalEncoding(d_model, dropout, max_len)

		self.layers = nn.ModuleList([EncoderLayer(d_model, heads, dropout) for _ in range(N)])

		# self.dropout = nn.Dropout(dropout)

		self.device = device

		self.scale = math.sqrt(d_model)

	def forward(self, src, src_mask):
		# src = [batch_size, src_len]
		# src_mask = [batch_size, 1, 1, src_len]
		# bs, src_len = src.shape

		# pos = torch.arange(0, src_len)[None].repeat(bs, 1).to(self.device)
		# # pos = [batch_size, src_len]
		#
		# x = self.dropout(self.tok_embed(src) * self.scale + self.pos_embed(pos))
		# # x = [batch_size, src_len, d_model]

		x = self.pe(self.tok_embed(src) * self.scale)
		# x = [batch_size, src_len, d_model]

		for layer in self.layers:
			x = layer(x, src_mask)
		# x = [batch_size, src_len, d_model]

		return x


class EncoderLayer(nn.Module):
	def __init__(self, d_model, heads, dropout):
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
