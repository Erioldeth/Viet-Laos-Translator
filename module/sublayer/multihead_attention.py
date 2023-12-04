import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
	def __init__(self, heads, d_model, dropout=0.1):
		super().__init__()

		assert d_model % heads == 0

		self.d_model = d_model
		self.h = heads
		self.d_k = d_model // heads

		self.WQ = nn.Linear(d_model, d_model)
		self.WK = nn.Linear(d_model, d_model)
		self.WV = nn.Linear(d_model, d_model)

		self.WO = nn.Linear(d_model, d_model)

		self.dropout = nn.Dropout(dropout)

	def forward(self, q, k, v, mask):
		# q = [batch_size, q_len, d_model]
		# k = [batch_size, k_len, d_model]
		# v = [batch_size, v_len, d_model]
		bs = len(q)

		q = self.WQ(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
		k = self.WK(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
		v = self.WV(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
		# q = [batch_size, heads, q_len, d_k]
		# k = [batch_size, heads, k_len, d_k]
		# v = [batch_size, heads, v_len, d_k]

		score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
		# score = [batch_size, heads, q_len, k_len]

		score = score.masked_fill(mask == 0, -1e9)

		attn = torch.softmax(score, dim=-1)
		# attn = [batch_size, heads, q_len, k_len]

		x = torch.matmul(self.dropout(attn), v)
		# x = [batch_size, heads, q_len, d_k]

		x = x.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
		# x = [batch_size, q_len, d_model]

		return self.WO(x), attn
