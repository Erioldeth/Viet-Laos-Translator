import math

import torch
import torch.nn as nn

from module.sublayer import *


class Decoder(nn.Module):
	def __init__(self, vocab_size, d_model, N, heads, dropout, max_seq_length=200):
		super().__init__()
		self.N = N

		self.tok_embed = nn.Embedding(vocab_size, d_model)
		self.pos_embed = nn.Embedding(max_seq_length, d_model)

		self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(N)])

		self.out = nn.Linear(d_model, vocab_size)

		self.dropout = nn.Dropout(dropout)

		# TODO: check use of this
		self._max_seq_length = max_seq_length

	def forward(self, trg, memory, src_mask, trg_mask):
		bs, trg_len = trg.shape

		pos = torch.arange(0, trg_len).unsqueeze(0).repeat(bs, 1)

		x = self.dropout(self.tok_embed(trg) * math.sqrt(self.d_model) + self.pos_embed(pos))

		for layer in self.layers:
			x, attn = layer(x, memory, src_mask, trg_mask)

		return self.out(x), attn


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
		trg_sa, _ = self.self_attn(trg, trg, trg, trg_mask)

		trg = self.sa_norm(trg + self.dropout(trg_sa))

		trg_ea, attn = self.enc_attn(trg, memory, memory, src_mask)

		trg = self.enc_norm(trg + self.dropout(trg_ea))

		trg_ff = self.ff(trg)

		trg = self.ff_norm(trg + self.dropout(trg_ff))

		return trg, attn
