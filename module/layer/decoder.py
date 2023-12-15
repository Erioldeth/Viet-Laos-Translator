import math

import torch.nn as nn

from module.sublayer import *


class Decoder(nn.Module):
	def __init__(self, output_dim, d_model, N, heads, dropout, max_len, device):
		super().__init__()

		self.tok_embed = nn.Embedding(output_dim, d_model)
		self.pe = PositionalEncoding(d_model, dropout, max_len)

		self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(N)])

		self.norm = nn.LayerNorm(d_model)

		self.out = nn.Linear(d_model, output_dim)

		self.device = device

		self.scale = math.sqrt(d_model)

	def forward(self, trg, memory, src_mask, trg_mask):
		# trg = [batch_size, trg_len]
		# memory = [batch_size, src_len, d_model]
		# src_mask = [batch_size, 1, 1, src_len]
		# trg_mask = [batch_size, 1, trg_len, trg_len]

		x = self.pe(self.tok_embed(trg) * self.scale)
		# x = [batch_size, trg_len, d_model]

		attn = None
		for layer in self.layers:
			x, attn = layer(x, memory, src_mask, trg_mask)
		# attn = [batch_size, heads, trg_len, src_len]

		x = self.out(self.norm(x))
		# x = [batch_size, trg_len, output_dim]

		return x, attn


class DecoderLayer(nn.Module):
	def __init__(self, d_model, heads, dropout):
		super().__init__()
		self.norm = nn.LayerNorm(d_model)

		self.self_attn = MultiHeadAttention(heads, d_model, dropout)
		self.enc_attn = MultiHeadAttention(heads, d_model, dropout)
		self.ff = FeedForward(d_model, dropout)

		self.dropout = nn.Dropout(dropout)

	def forward(self, trg, memory, src_mask, trg_mask):
		# trg = [batch_size, trg_len, d_model]
		# memory = [batch_size, src_len, d_model]
		# src_mask = [batch_size, 1, 1, src_len]
		# trg_mask = [batch_size, 1, trg_len, trg_len]

		x = self.norm(trg)
		x, _ = self.self_attn(x, x, x, trg_mask)
		trg = trg + self.dropout(x)

		x = self.norm(trg)
		x, attn = self.enc_attn(x, memory, memory, src_mask)
		# attn = [batch_size, heads, trg_len, src_len]
		trg = trg + self.dropout(x)

		x = self.norm(trg)
		x = self.ff(x)
		trg = trg + self.dropout(x)

		return trg, attn
