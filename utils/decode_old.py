import numpy as np
import torch
from torch.autograd import Variable


def no_peeking_mask(size, device):
	"""
	Tạo mask được sử dụng trong decoder để lúc dự đoán trong quá trình huấn luyện
	mô hình không nhìn thấy được các từ ở tương lai
	"""
	np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
	np_mask = Variable(torch.from_numpy(np_mask) == 0)
	np_mask = np_mask.to(device)

	return np_mask


def create_masks(src, trg, src_pad, trg_pad, device):
	src_mask = (src != src_pad).unsqueeze(-2)

	if trg is not None:
		trg_mask = (trg != trg_pad).unsqueeze(-2)
		size = trg.size(1)  # get seq_len for matrix
		np_mask = no_peeking_mask(size, device)
		if trg.is_cuda:
			np_mask.cuda()
		trg_mask = trg_mask & np_mask

	else:
		trg_mask = None
	return src_mask, trg_mask
