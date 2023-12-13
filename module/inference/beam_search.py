import gc
import operator

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class BeamSearch:
	def __init__(self, model, max_len, device, beam_size, length_normalize):
		self.model = model
		self.SRC, self.TRG = model.SRC, model.TRG
		self.max_len = max_len

		self.beam_size = beam_size
		self._length_norm = length_normalize

		self.device = device

	def transl_batch(self, batch: list[str], src_size_limit) -> list[str]:
		padded_batch, tokenized_batch = self.preprocess_batch(batch, src_size_limit)
		# padded_batch = [batch_size, max_src_len]
		# tokenized_batch = [batch_size, *src_len]

		translated_batch = self.search(padded_batch, tokenized_batch)

		return [' '.join(token_list) for token_list in translated_batch]

	def preprocess_batch(self, batch: list[str], src_size_limit) -> tuple[Tensor, list[list[str]]]:
		tokenized_batch = [self.SRC.preprocess(s)[:src_size_limit] for s in batch]
		# preprocessed_batch = [batch_size, *src_len]

		indexed_batch = [torch.tensor([self.SRC.vocab.stoi[t] for t in s], dtype=torch.long) for s in tokenized_batch]
		# indexed_batch = [batch_size, *src_len]

		padded_batch = pad_sequence(indexed_batch, True, self.SRC.vocab.stoi['<pad>'])
		# batch = [batch_size, max_src_len]

		return padded_batch, tokenized_batch

	def search(self, batch: Tensor, src_tokens: list[list[str]]) -> list[list[str]]:
		model = self.model
		device = self.device
		beam_size = self.beam_size
		trg_eos_idx = self.TRG.vocab.stoi['<eos>']
		batch_size = len(batch)

		hypotheses, memory, log_scores = self.init_search(batch)
		# hypotheses = [batch_size * beam_size, max_len]
		# memory = [batch_size * beam_size, src_len, d_model]
		# log_scores = [batch_size * beam_size, 1]

		src_mask = torch.repeat_interleave((batch != model.src_pad_idx)[:, None, None], beam_size, 0).to(device)
		# src_mask = [batch_size * beam_size, 1, 1, src_len]

		all_eos = torch.full([batch_size * beam_size], trg_eos_idx, dtype=torch.long).to(device)
		# all_eos = [batch_size * beam_size]

		attn = None
		for i in range(2, self.max_len):
			gc.collect()
			torch.cuda.empty_cache()

			trg_mask = torch.tril(torch.ones(i, i)).to(device, dtype=torch.bool)

			output, attn = model.decoder(hypotheses[:, :i], memory, src_mask, trg_mask)
			# output = [batch_size * beam_size, i, output_dim]
			# attn = [batch_size * beam_size, heads, i, src_len]

			hypotheses, log_scores = self.get_best(hypotheses, output, log_scores, i)

			if torch.equal(hypotheses[:, i], all_eos):
				break

		hypotheses = hypotheses.view(batch_size, self.beam_size, -1)
		log_scores = log_scores.view(batch_size, self.beam_size)
		# hypotheses = [batch_size, beam_size, trg_len]
		# log_scores = [batch_size, beam_size]

		transl_tokens = lambda tokens: [self.TRG.vocab.itos[token] for token in tokens[1:self._length(tokens)]]
		translated_tokens = np.array([[transl_tokens(beam) for beam in beams] for beams in hypotheses], dtype=object)
		# translated_tokens = [batch_size, beam_size, *trg_len]

		translated_tokens = self.replace_unknown(translated_tokens, src_tokens, attn)
		# translated_tokens = [batch_size, beam_size, *trg_len]

		if self._length_norm is not None:
			lengths = np.array([[self._length(beam) for beam in beams] for beams in hypotheses.cpu()])
			# lengths = [batch_size, beam_size]

			penalized_probs = log_scores.detach().cpu().numpy() / (((lengths + 5) / 6) ** self._length_norm)
			# penalized_probs = [batch_size, beam_size]

			indices = np.argsort(penalized_probs)[:, ::-1]
			# indices = [batch_size, beam_size]

			translated_tokens = np.array([beam[idx] for beam, idx in zip(translated_tokens, indices)])

		return translated_tokens[:, 0].tolist()

	def init_search(self, batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
		model = self.model
		device = self.device
		beam_size = self.beam_size
		trg_sos_idx = self.TRG.vocab.stoi['<sos>']
		batch_size = len(batch)

		src = batch.to(device)
		src_mask = (src != model.src_pad_idx)[:, None, None].to(device)
		memory = model.encoder(src, src_mask)
		# src = [batch_size, src_len]
		# src_mask = [batch_size, 1, 1, src_len]
		# memory = [batch_size, src_len, d_model]

		trg = torch.full([batch_size, 1], trg_sos_idx).to(device, dtype=torch.long)
		trg_mask = torch.tril(torch.ones(1, 1)).to(device, dtype=torch.bool)
		output, _ = model.decoder(trg, memory, src_mask, trg_mask)
		# trg = [batch_size, 1]
		# trg_mask = [1, 1]
		# output = [batch_size, 1, output_dim]

		softmax_probs, indices = torch.softmax(output, dim=-1)[:, -1].topk(beam_size)
		# softmax_probs = [batch_size, beam_size]
		# indices = [batch_size, beam_size]

		log_scores = torch.log(softmax_probs).view(-1, 1)
		# log_scores = [batch_size * beam_size, 1]

		hypotheses = torch.zeros(batch_size * beam_size, self.max_len).to(device, dtype=torch.long)
		# hypotheses = [batch_size * beam_size, max_len]
		hypotheses[:, 0] = trg_sos_idx
		hypotheses[:, 1] = indices.view(-1)

		memory = torch.repeat_interleave(memory, beam_size, 0)
		# memory = [batch_size * beam_size, src_len, d_model]

		return hypotheses, memory, log_scores

	def get_best(self, hypotheses: Tensor, output: Tensor, log_scores: Tensor, i) -> tuple[Tensor, Tensor]:
		device = self.device
		beam_size = self.beam_size
		trg_eos_idx = self.TRG.vocab.stoi['<eos>']
		width = len(hypotheses)
		batch_size = width // beam_size

		probs, indices = torch.softmax(output, dim=-1)[:, -1].topk(beam_size)
		# probs = [width, beam_size]
		# indices = [width, beam_size]

		repl_probs = torch.full([width, beam_size], 1e-100, dtype=torch.float).to(device)
		repl_probs[:, 0] = 1
		repl_indices = torch.full([width, beam_size], -1).to(device)
		repl_indices[:, 0] = trg_eos_idx

		is_eos = torch.repeat_interleave((hypotheses[:, i - 1] == trg_eos_idx)[:, None], beam_size, 1)
		# is_eos = [width, beam_size]

		probs, indices = torch.where(is_eos, repl_probs, probs), torch.where(is_eos, repl_indices, indices)
		# probs = [width, beam_size]
		# indices = [width, beam_size]

		k_probs, k_indices = (torch.log(probs) + log_scores).to(device).view(batch_size, -1).topk(beam_size)
		# k_probs = [batch_size, beam_size]
		# k_indices = [batch_size, beam_size]

		row = (k_indices // beam_size + torch.arange(0, batch_size * beam_size, beam_size)[:, None].to(device)).view(-1)
		col = (k_indices % beam_size).view(-1)
		# row = [batch_size * beam_size]
		# col = [batch_size * beam_size]

		hypotheses[:, :i] = hypotheses[row, :i]
		hypotheses[:, i] = indices[row, col]

		return hypotheses, k_probs.view(-1, 1)

	def replace_unknown(self, translated_tokens: ndarray, src_tokens: list[list[str]], attn: Tensor) -> ndarray:
		used_attention = attn.sum(1)
		# used_attention = [batch_size * beam_size, trg_len, src_len]

		best_src_indices = torch.argmax(used_attention, dim=-1).cpu().numpy()
		# select_id_src = [batch_size * beam_size, trg_len]

		repl_tokens = np.array([operator.itemgetter(*src_indices)(src_tokens[bb_i // self.beam_size])
		                        for bb_i, src_indices in enumerate(best_src_indices)],
		                       dtype=object)
		# repl_tokens = [batch_size * beam_size, trg_len]

		orig_tokens = translated_tokens.ravel()
		# orig_tokens = [batch_size * beam_size, *trg_len]

		replaced = np.array([[ori if ori != '<unk>' else rpl
		                      for ori, rpl in zip(orig, repl)]
		                     for orig, repl in zip(orig_tokens, repl_tokens)],
		                    dtype=object)

		return replaced.reshape(translated_tokens.shape)

	def _length(self, tokens: Tensor):
		eos, = (tokens == self.TRG.vocab.stoi['<eos>']).nonzero(as_tuple=True)
		return len(tokens) if len(eos) == 0 else eos[0]
