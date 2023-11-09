import copy

import torch.nn as nn


def get_clones(module, N, keep_module=True):
	if keep_module and N >= 1:
		# create N-1 copies in addition to the original
		return nn.ModuleList([module] + [copy.deepcopy(module) for _ in range(N - 1)])
	else:
		# create N new copy
		return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
