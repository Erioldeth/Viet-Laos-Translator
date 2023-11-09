import torch.nn as nn
import torch.nn.functional as functional


class FeedForward(nn.Module):
	"""A two-hidden-linear feedforward layer that can activate and dropout its transition state"""

	def __init__(self, d_model, d_ff=2048, internal_activation=functional.relu, dropout=0.1):
		super().__init__()
		self.linear_1 = nn.Linear(d_model, d_ff)
		self.dropout = nn.Dropout(dropout)
		self.linear_2 = nn.Linear(d_ff, d_model)

		self.internal_activation = internal_activation

	def forward(self, x):
		x = self.dropout(self.internal_activation(self.linear_1(x)))
		x = self.linear_2(x)
		return x
