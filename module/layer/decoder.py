import math

import torch
import torch.nn as nn

from module.sublayer import *


class Decoder(nn.Module):
	def __init__(self, output_dim, d_model, N, heads, dropout, max_len):
		super().__init__()

		self.tok_embed = nn.Embedding(output_dim, d_model)
		self.pos_embed = nn.Embedding(max_len, d_model)

		self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(N)])

		self.out = nn.Linear(d_model, output_dim)

		self.dropout = nn.Dropout(dropout)

		self.scale = math.sqrt(d_model)

	def forward(self, trg, memory, src_mask, trg_mask):
		# trg = [batch_size, trg_len]
		# memory = [batch_size, src_len, d_model]
		# src_mask = [batch_size, 1, 1, src_len]
		# trg_mask = [batch_size, 1, trg_len, trg_len]
		bs, trg_len = trg.shape

		pos = torch.arange(0, trg_len)[None].repeat(bs, 1)
		# pos = [batch_size, trg_len]

		x = self.dropout(self.tok_embed(trg) * self.scale + self.pos_embed(pos))
		# x = [batch_size, trg_len, d_model]

		attn = None
		for layer in self.layers:
			x, attn = layer(x, memory, src_mask, trg_mask)
		# x = [batch_size, trg_len, d_model]
		# attn = [batch_size, heads, trg_len, src_len]

		x = self.out(x)
		# x = [batch_size, trg_len, output_dim]

		return x, attn


class DecoderLayer(nn.Module):
	def __init__(self, d_model, heads, dropout=0.1):
		super().__init__()

		self.sa_norm = nn.LayerNorm(d_model)
		self.enc_norm = nn.LayerNorm(d_model)
		self.ff_norm = nn.LayerNorm(d_model)

		self.self_attn = MultiHeadAttention(heads, d_model, dropout)
		self.enc_attn = MultiHeadAttention(heads, d_model, dropout)
		self.ff = FeedForward(d_model, dropout)

		self.dropout = nn.Dropout(dropout)

	def forward(self, trg, memory, src_mask, trg_mask):
		# trg = [batch_size, trg_len, d_model]
		# memory = [batch_size, src_len, d_model]
		# src_mask = [batch_size, 1, 1, src_len]
		# trg_mask = [batch_size, 1, trg_len, trg_len]
		trg_sa, _ = self.self_attn(trg, trg, trg, trg_mask)

		trg = self.sa_norm(trg + self.dropout(trg_sa))
		# trg = [batch_size, trg_len, d_model]

		trg_ea, attn = self.enc_attn(trg, memory, memory, src_mask)

		trg = self.enc_norm(trg + self.dropout(trg_ea))
		# trg = [batch_size, trg_len, d_model]

		trg_ff = self.ff(trg)

		trg = self.ff_norm(trg + self.dropout(trg_ff))
		# trg = [batch_size, trg_len, d_model]
		# attn = [batch_size, heads, trg_len, src_len]

		return trg, attn
