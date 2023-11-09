import torch.nn as nn

from module.sublayer import *
from support import get_clones


class Decoder(nn.Module):
	"""A wrapper that receive the encoder outputs, run through the decoder process for a determined input
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
		self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
		self.norm = Norm(d_model)

		self._max_seq_length = max_seq_length

	def forward(self, trg, memory, src_mask, trg_mask, output_attention=False):
		"""Accepts a batch of indexed tokens and the encoding outputs, return the decoded values.
		Args:
			trg: input Tensor of [batch_size, trg_len]
			memory: output of Encoder [batch_size, src_len, d_model]
			src_mask: the padding mask, [batch_size, 1, src_len]
			trg_mask: the no-peeking mask, [batch_size, trg_len, trg_len]
			output_attention: if set, output a list containing used attention
		Returns:
			the decoded values [batch_size, trg_len, d_model]
			if available, list of N (self-attention, attention) calculated.
			They are in form of [batch_size, heads, trg_len, trg/src_len]
		"""
		x = self.embed(trg)
		x = self.pe(x)

		attentions = [None] * self.N
		for i in range(self.N):
			x, attn = self.layers[i](x, memory, src_mask, trg_mask)
			attentions[i] = attn
		x = self.norm(x)
		return x if not output_attention else (x, attentions)


class DecoderLayer(nn.Module):
	def __init__(self, d_model, heads, dropout=0.1):
		"""
		A layer of the decoder.
		Contain a self-attention that accept no-peeking mask and a normal attention that accept padding mask
		Args:
			d_model: the inner dimension size of the layer
			heads: number of heads used in the attention
			dropout: applied dropout value during training
		"""
		super().__init__()
		self.norm_1 = Norm(d_model)
		self.norm_2 = Norm(d_model)
		self.norm_3 = Norm(d_model)

		self.dropout_1 = nn.Dropout(dropout)
		self.dropout_2 = nn.Dropout(dropout)
		self.dropout_3 = nn.Dropout(dropout)

		self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
		self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
		self.ff = FeedForward(d_model, dropout=dropout)

	def forward(self, x, memory, src_mask, trg_mask):
		"""Run the decoding layer
		Args:
			x: the input (either embedding values or previous layer output), should be in shape [batch_size, trg_len, d_model]
			memory: the outputs of the encoding section, used for normal attention. [batch_size, src_len, d_model]
			src_mask: the padding mask for the memory, [batch_size, 1, src_len]
			trg_mask: the no-peeking mask for the decoder, [batch_size, trg_len, trg_len]
		Return:
			an output that have the same shape as input, [batch_size, trg_len, d_model]
			the self-attention and normal attention received
			[batch_size, head, trg_len, trg_len] & [batch_size, head, trg_len, src_len]
		"""
		x2 = self.norm_1(x)
		# Self-attention
		x_sa, sa = self.attn_1(x2, x2, x2, trg_mask)
		x = x + self.dropout_1(x_sa)
		x2 = self.norm_2(x)
		# Normal multi-head attention
		x_na, na = self.attn_2(x2, memory, memory, src_mask)
		x = x + self.dropout_2(x_na)
		x2 = self.norm_3(x)
		x = x + self.dropout_3(self.ff(x2))
		return x, (sa, na)
