import torch.nn as nn
from torch.nn.functional import gelu


class FeedForward(nn.Module):
	def __init__(self, d_model, dropout=0.1):
		super().__init__()

		self.W1 = nn.Linear(d_model, 2048)
		self.W2 = nn.Linear(2048, d_model)

		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		x = self.dropout(gelu(self.W1(x)))
		return self.W2(x)
