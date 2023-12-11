import gc
import io
import math
import time

import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from model.save import save_model
from module import *


class Transformer(nn.Module):
	def __init__(self, mode, model_dir, config_path, device):
		super().__init__()

		self.device = device

		print('Loading config ...')
		with io.open(config_path, 'r', encoding='utf-8') as stream:
			opt = self.config = yaml.full_load(stream)

		print('Loading data ...')
		data = opt['data']
		train_path = data['train_data_location']
		valid_path = data['valid_data_location']
		lang_tuple = data['src_lang'], data['trg_lang']
		self.loader = Loader(train_path, valid_path, lang_tuple, opt)

		print('Building vocab ...')
		self.SRC, self.TRG = self.fields = self.loader.build_fields(model_dir)

		print('Creating iterator ...')
		match mode:
			case 'train':
				self.train_iter, self.valid_iter = self.loader.create_iterator(self.fields, model_dir, self.device)
			case 'infer':
				self.loader.build_vocab(self.fields, model_dir)

		self.src_pad_idx, self.trg_pad_idx = self.SRC.vocab.stoi['<pad>'], self.TRG.vocab.stoi['<pad>']

		print('Building encoder and decoder ...')
		d_model, N, heads, dropout = opt['d_model'], opt['n_layers'], opt['heads'], opt['dropout']

		self.encoder = Encoder(len(self.SRC.vocab), d_model, N, heads, dropout, 200, self.device)
		self.decoder = Decoder(len(self.TRG.vocab), d_model, N, heads, dropout, 200, self.device)

	def forward(self, src, trg):
		# src = [batch_size, src_len]
		# trg = [batch_size, trg_len]

		src_mask, trg_mask = self.make_masks(src, trg)
		# src_mask = [batch_size, 1, 1, src_len]
		# trg_mask = [batch_size, 1, trg_len, trg_len]

		memory = self.encoder(src, src_mask)
		# memory = [batch_size, src_len, d_model]

		output, _ = self.decoder(trg, memory, src_mask, trg_mask)
		# output = [batch_size, trg_len, output_dim]

		return output

	def make_masks(self, src, trg):
		# src = [batch_size, src_len]
		# trg = [batch_size, trg_len]
		device = self.device

		src_mask = (src != self.src_pad_idx)[:, None, None]
		# src_mask = [batch_size, 1, 1, src_len]

		trg_pad_mask = (trg != self.trg_pad_idx)[:, None, None]
		# trg_pad_mask = [batch_size, 1, 1, trg_len]
		trg_len = trg.shape[1]
		trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len, device=device)).bool()
		# trg_sub_mask = [trg_len, trg_len]
		trg_mask = trg_pad_mask & trg_sub_mask
		# trg_mask = [batch_size, 1, trg_len, trg_len]

		return src_mask, trg_mask

	def perform_training(self, criterion, optimizer, scheduler):
		self.train()

		total_loss = 0.0

		for batch in self.train_iter:
			src, trg = batch.src, batch.trg
			# src = [batch_size, src_len]
			# trg = [batch_size, trg_len]

			output = self(src, trg[:, :-1])
			# output = [batch_size, trg_len - 1, output_dim]

			output_dim = output.shape[-1]

			output = output.contiguous().view(-1, output_dim)
			trg = trg[:, 1:].contiguous().view(-1)
			# output = [batch_size * (trg len - 1), output_dim]
			# trg = [batch_size * (trg_len - 1)]

			loss = criterion(output, trg)

			optimizer.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(self.parameters(), 1)
			optimizer.step()
			scheduler.step()

			total_loss += loss.item()

		return total_loss / len(self.train_iter)

	def perform_validating(self, criterion):
		self.eval()

		total_loss = 0.0

		with torch.no_grad():
			for batch in self.valid_iter:
				src, trg = batch.src, batch.trg
				# src = [batch_size, src_len]
				# trg = [batch_size, trg_len]

				output = self(src, trg[:, :-1])
				# output = [batch_size, trg_len - 1, output_dim]

				output_dim = output.shape[-1]

				output = output.contiguous().view(-1, output_dim)
				trg = trg[:, 1:].contiguous().view(-1)
				# output = [batch_size * (trg_len - 1), output_dim]
				# trg = [batch_size * (trg_len - 1)]

				loss = criterion(output, trg)

				total_loss += loss.item()

		return total_loss / len(self.valid_iter)

	def run_train(self, model_dir):
		print(f'src vocab size = {len(self.SRC.vocab)}')
		print(f'trg vocab size = {len(self.TRG.vocab)}')
		print(f'Encoder: {sum([p.numel() for p in self.encoder.parameters() if p.requires_grad])} parameters')
		print(f'Decoder: {sum([p.numel() for p in self.decoder.parameters() if p.requires_grad])} parameters')
		print(f'Starting training on {self.device}')
		print('Performing training...')
		print('=' * 50)

		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

		opt = self.config

		optimizer = Adam(self.parameters(), opt['lr'], **opt['optimizer_params'])
		d_model, n_warmup_steps = opt['d_model'], opt['n_warmup_steps']
		lr_lambda = lambda step: d_model ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * n_warmup_steps ** (-1.5))
		scheduler = LambdaLR(optimizer, lr_lambda)
		criterion = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx, label_smoothing=opt['label_smoothing'])

		best_valid_loss = float('inf')

		for epoch in range(opt['epochs']):
			gc.collect()
			torch.cuda.empty_cache()
			start = time.time()

			train_loss = self.perform_training(criterion, optimizer, scheduler)
			valid_loss = self.perform_validating(criterion)

			elapsed_time = time.time() - start
			print(f'Epoch: {epoch + 1:02} - {elapsed_time // 60}m{elapsed_time % 60}s')
			print(f'\tTrain Loss/PPL: {train_loss:7.3f} / {math.exp(train_loss):7.3f}')
			print(f'\tVal   Loss/PPL: {valid_loss:7.3f} / {math.exp(valid_loss):7.3f}')
			print('-' * 50)

			if valid_loss < best_valid_loss:
				best_valid_loss = valid_loss
				save_model(self, model_dir)

	def run_infer(self, features_file, predictions_file):
		self.eval()

		opt = self.config
		input_max_len = opt['input_max_length']
		infer_max_len = opt['infer_max_length']
		batch_size = opt['infer_batch_size']

		print('Building decode strategy ...')
		decode_strategy = BeamSearch(self, infer_max_len, self.device, **opt['decode_strategy_kwargs'])

		print(f'Reading features file from {features_file}...')
		with io.open(features_file, 'rt', encoding='utf-8') as file:
			inputs = [line.strip() for line in file.readlines()]

		print('Performing inference ...')
		start = time.time()

		results = '\n'.join([translated_sentence
		                     for batch in [inputs[i: i + batch_size] for i in range(0, len(inputs), batch_size)]
		                     for translated_sentence in decode_strategy.transl_batch(batch, input_max_len)])

		elapsed_time = time.time() - start
		print(f'Inference done, cost {elapsed_time // 60}m{elapsed_time % 60}s')

		print(f'Writing results to {predictions_file} ...')
		with io.open(predictions_file, 'w', encoding='utf-8') as file:
			file.write(results)

		print('All done!')
