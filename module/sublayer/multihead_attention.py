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

	def forward(self, q, k, v, mask=None):
		bs = q.shape[0]

		q = self.WQ(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
		k = self.WK(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
		v = self.WV(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

		scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

		if mask is not None:
			scores = scores.masked_fill(mask == 0, -1e9)

		attn = torch.softmax(scores, dim=-1)

		value = torch.matmul(self.dropout(attn), v)
		value = value.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

		return self.WO(value), attn
