import math

import torch
import torch.nn as nn
import torch.nn.functional as functional


class MultiHeadAttention(nn.Module):
	def __init__(self, heads, d_model, dropout=0.1):
		super().__init__()
		assert d_model % heads == 0

		self.d_model = d_model
		self.d_k = d_model // heads
		self.h = heads

		# three casting linear layer for query/key.value
		self.q_linear = nn.Linear(d_model, d_model)
		self.k_linear = nn.Linear(d_model, d_model)
		self.v_linear = nn.Linear(d_model, d_model)

		self.dropout = nn.Dropout(dropout)
		self.out = nn.Linear(d_model, d_model)

	def forward(self, q, k, v, mask=None):
		"""
		Args:
			q/k/v: query/key/value, should all be [batch_size, sequence_length, d_model].
			Only differ in decode attention, where q is trg_len and k/v is src_len
			mask: either [batch_size, 1, src_len] or [batch_size, trg_len, trg_len].
			The last two dimensions must match or are broadcastable.
		Returns:
			the value of the attention process, [batch_size, sequence_length, d_model].
			The used attention, [batch_size, q_length, k_v_length]
		"""
		bs = q.shape[0]
		q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
		k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
		v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

		value, attn = self.attention(q, k, v, mask, self.dropout)
		concat = value.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
		output = self.out(concat)
		return output, attn

	def attention(self, q, k, v, mask=None, dropout=None):
		"""Calculate the attention and output the attention & value
		Args:
			q / k / v: query/key/value already transformed, should all be [batch_size, heads, sequence_length, d_k.
			Only differ in decode attention, where q is trg_len and k/v is src_len
			mask: either [batch_size, 1, src_len] or [batch_size, trg_len, trg_len].
			The last two dimensions must match or are broadcastable.
		Returns:
			the attentionized but raw values [batch_size, head, seq_length, d_k]
			the attention calculated [batch_size, heads, sequence_length, sequence_length]
		"""

		scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

		if mask is not None:
			# add a dimension to account for head
			mask = mask.unsqueeze(1)
			scores = scores.masked_fill(mask == 0, -1e9)

		# softmax the padding/peeking masked attention
		scores = functional.softmax(scores, dim=-1)

		if dropout is not None:
			scores = dropout(scores)

		output = torch.matmul(scores, v)
		return output, scores
