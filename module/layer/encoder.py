import torch.nn as nn

from module.sublayer import *
from support import get_clones


class Encoder(nn.Module):
	"""A wrapper that embed, positional encode, and self-attention encode the inputs.
	Args:
		vocab_size: the size of the vocab. Used for embedding
		d_model: the inner dim of the module
		N: number of layers used
		heads: number of heads used in the attention
		dropout: applied dropout value during training
		max_seq_length: the maximum length value used for this encoder. Needed for PositionalEncoder, due to caching
	"""

	def __init__(self, vocab_size, d_model, N, heads, dropout, max_seq_length=200):
		super().__init__()
		self.N = N
		self.embed = nn.Embedding(vocab_size, d_model)
		self.pe = PositionalEncoder(d_model, dropout=dropout, max_seq_length=max_seq_length)
		self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
		self.norm = Norm(d_model)

		self._max_seq_length = max_seq_length

	def forward(self, src, src_mask, output_attention=False, seq_length_check=False):
		"""Accepts a batch of indexed tokens, return the encoded values.
		Args:
			src: int Tensor of [batch_size, src_len]
			src_mask: the padding mask, [batch_size, 1, src_len]
			output_attention: if set, output a list containing used attention
			seq_length_check: if set, automatically trim the input if it goes past the expected sequence length.
		Returns:
			the encoded values [batch_size, src_len, d_model]
			if available, list of N (self-attention) calculated. They are in form of [batch_size, heads, src_len, src_len]
		"""
		if seq_length_check and src.shape[1] > self._max_seq_length:
			src = src[:, :self._max_seq_length]
			src_mask = src_mask[:, :, :self._max_seq_length]

		x = self.embed(src)
		x = self.pe(x)
		attentions = [None] * self.N

		for i in range(self.N):
			x, attn = self.layers[i](x, src_mask)
			attentions[i] = attn

		x = self.norm(x)
		return x if not output_attention else (x, attentions)


class EncoderLayer(nn.Module):
	def __init__(self, d_model, heads, dropout=0.1):
		"""A layer of the encoder. Contain a self-attention accepting padding mask
		Args:
			d_model: the inner dimension size of the layer
			heads: number of heads used in the attention
			dropout: applied dropout value during training
		"""
		super().__init__()

		self.norm_1 = Norm(d_model)
		self.norm_2 = Norm(d_model)

		self.dropout_1 = nn.Dropout(dropout)
		self.dropout_2 = nn.Dropout(dropout)

		self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
		self.ff = FeedForward(d_model, dropout=dropout)

	def forward(self, x, src_mask):
		"""Run the encoding layer
		Args:
			x: the input (either embedding values or previous layer output), should be in shape [batch_size, src_len, d_model]
			src_mask: the padding mask, should be [batch_size, 1, src_len]
		Return:
			an output that have the same shape as input, [batch_size, src_len, d_model]
			the attention used [batch_size, src_len, src_len]
		"""
		x2 = self.norm_1(x)
		# Self attention only
		x_sa, sa = self.attn(x2, x2, x2, src_mask)
		x = x + self.dropout_1(x_sa)

		x2 = self.norm_2(x)
		x = x + self.dropout_2(self.ff(x2))

		return x, sa
