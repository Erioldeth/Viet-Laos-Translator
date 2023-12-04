import io
import math
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from config import *
from model.save import save_model
from module import *


class Transformer(nn.Module):

	def __init__(self, mode, model_dir, config_path):
		super().__init__()

		assert mode in ['train', 'infer'], f'Unknown mode: {mode}'
		assert model_dir is not None and config_path is not None, 'Missing model_dir and config_path'

		opt = self.config = Config(config_path)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		data_opt = opt['data']
		self.loader = Loader(train_path=data_opt['train_data_location'],
		                     valid_path=data_opt['valid_data_location'],
		                     lang_tuple=(data_opt['src_lang'], data_opt['trg_lang']),
		                     option=opt)

		self.SRC, self.TRG = self.fields = self.loader.build_fields()

		match mode:
			case 'train':
				self.train_iter, self.valid_iter = self.loader.create_iterator(self.fields, model_dir, self.device)
			case 'infer':
				self.loader.build_vocab(self.fields, model_dir)

		self.src_pad_idx = self.SRC.vocab.stoi['<pad>']
		self.trg_pad_idx = self.TRG.vocab.stoi['<pad>']

		d_model, N, heads, dropout = opt['d_model'], opt['n_layers'], opt['heads'], opt['dropout']
		train_ignore_len = opt['train_max_length']
		input_max_len = opt['input_max_length']
		infer_max_len = opt['max_length']
		encoder_max_len = max(input_max_len, train_ignore_len)
		decoder_max_len = max(infer_max_len, train_ignore_len)

		self.encoder = Encoder(len(self.SRC.vocab), d_model, N, heads, dropout, encoder_max_len)
		self.decoder = Decoder(len(self.TRG.vocab), d_model, N, heads, dropout, decoder_max_len)

		self.out = nn.Linear(d_model, len(self.TRG.vocab))

		decode_strategy_kwargs = opt['decode_strategy_kwargs']
		self.decode_strategy = BeamSearch(self, infer_max_len, self.device, **decode_strategy_kwargs)

		self.to(self.device)

	def forward(self, src, trg):
		# src = [batch_size, src_len]
		# trg = [batch_size, trg_len]

		src_mask, trg_mask = self.make_masks(src, trg)
		# src_mask = [batch_size, 1, 1, src_len]
		# trg_mask = [batch_size, 1, trg_len, trg_len]

		memory = self.encoder(src, src_mask)
		# memory = [batch_size, src_len, d_model]

		output, attn = self.decoder(trg, memory, src_mask, trg_mask)
		# output = [batch_size, trg_len, output_dim]
		# attn = [batch_size, n_heads, trg_len, src_len]

		return output, attn

	def make_masks(self, src, trg):
		# src = [batch_size, src_len]
		# trg = [batch_size, trg_len]
		device = self.device

		src_mask = (src != self.src_pad_idx)[:, None, None].to(device)
		# src_mask = [batch_size, 1, 1, src_len]

		trg_pad_mask = (trg != self.trg_pad_idx)[:, None, None].to(device)
		# trg_pad_mask = [batch_size, 1, 1, trg_len]
		trg_len = trg.shape[1]
		trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).to(device, dtype=torch.bool)
		# trg_sub_mask = [trg_len, trg_len]
		trg_mask = trg_pad_mask & trg_sub_mask
		# trg_mask = [batch_size, 1, trg_len, trg_len]

		return src_mask, trg_mask

	def training(self, criterion, optimizer, scheduler):
		self.train()

		total_loss = 0.0

		for batch in self.train_iter:
			src, trg = batch.src, batch.trg
			# src = [batch_size, src_len]
			# trg = [batch_size, trg_len]

			output, _ = self(src, trg[:, :-1])
			# output = [batch_size, trg_len - 1, output_dim]

			output_dim = output.shape[-1]

			output = output.contiguous().view(-1, output_dim)
			trg = trg[:, 1:].contiguous().view(-1)
			# output = [batch_size * trg len - 1, output_dim]
			# trg = [batch_size * trg_len - 1]

			loss = criterion(output, trg)

			optimizer.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(self.parameters(), 1)
			optimizer.step()
			scheduler.step()

			total_loss += loss.item()

		return total_loss / len(self.train_iter)

	def validating(self, criterion):
		self.eval()

		total_loss = 0.0

		with torch.no_grad():
			for batch in self.valid_iter:
				src, trg = batch.src, batch.trg
				# src = [batch_size, src_len]
				# trg = [batch_size, trg_len]

				output, _ = self(src, trg[:, :-1])
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
		lr, optimizer_params = opt['lr'], opt['optimizer_params']
		d_model, n_warmup_steps = opt['d_model'], opt['n_warmup_steps']
		label_smoothing = opt['label_smoothing']

		optimizer = Adam(self.parameters(), lr, **optimizer_params)
		lr_lambda = lambda step: d_model ** (-0.5) * min(step ** (-0.5), step * n_warmup_steps ** (-1.5))
		scheduler = LambdaLR(optimizer, lr_lambda)
		criterion = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx, label_smoothing=label_smoothing)

		best_valid_loss = float('inf')

		for epoch in range(opt['epochs']):
			start = time.time()

			train_loss = self.training(criterion, optimizer, scheduler)
			valid_loss = self.validating(criterion)

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
		batch_size = opt['valid_batch_size']
		input_max_length = opt['input_max_length']

		print(f'Reading features file from {features_file}...')
		with io.open(features_file, 'r', encoding='utf-8') as file:
			inputs = [line.strip() for line in file.readlines()]

		print('Performing inference ...')
		start = time.time()

		results = '\n'.join([translated_sentence
		                     for batch in [inputs[i: i + batch_size] for i in range(0, len(inputs), batch_size)]
		                     for translated_sentence in self.decode_strategy.transl_batch(batch, input_max_length)])

		elapsed_time = time.time() - start
		print(f'Inference done, cost {elapsed_time // 60}m{elapsed_time % 60}s')

		print(f'Writing results to {predictions_file} ...')
		with io.open(predictions_file, 'w', encoding='utf-8') as file:
			file.write(results)

		print('All done!')
