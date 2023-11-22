import io
import time

import numpy as np
import torch
import torch.nn as nn
from torchtext.data import Field

import utils.save as saver
from config import *
from module.inference import strategies
from module.layer import *
from module.loader import Loader
from module.optim import optimizers, ScheduledOptim
from utils.logger import init_logger
from utils.loss import LabelSmoothingLoss
from utils.metric import bleu_batch_iter


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

		# TODO: improve preprocessing
		src_kwargs = {}
		trg_kwargs = {}
		field_kwargs = {
			'init_token': '<sos>',
			'eos_token': '<eos>',
			'lower': True,
			'batch_first': True
		}
		self.fields = self.SRC, self.TRG = Field(**src_kwargs, **field_kwargs), Field(**trg_kwargs, **field_kwargs)

		match mode:
			case 'train':
				self.train_iter, self.valid_iter = self.loader.create_iterator(self.fields, model_dir, self.device)
			case 'infer':
				self.loader.build_vocab(self.fields, model_dir)

		self.src_pad_idx = self.SRC.vocab.stoi['<pad>']
		self.trg_pad_idx = self.TRG.vocab.stoi['<pad>']

		src_vocab_size, trg_vocab_size = len(self.SRC.vocab), len(self.TRG.vocab)
		d_model, N, heads, dropout = opt['d_model'], opt['n_layers'], opt['heads'], opt['dropout']
		train_ignore_len = opt['train_max_length']
		input_max_len = opt['input_max_length']
		infer_max_len = opt['max_length']
		encoder_max_len = max(input_max_len, train_ignore_len)
		decoder_max_len = max(infer_max_len, train_ignore_len)

		self.encoder = Encoder(src_vocab_size, d_model, N, heads, dropout, encoder_max_len)
		self.decoder = Decoder(trg_vocab_size, d_model, N, heads, dropout, decoder_max_len)

		self.out = nn.Linear(d_model, trg_vocab_size)

		decode_strategy_cls = strategies[opt['decode_strategy']]
		decode_strategy_kwargs = opt['decode_strategy_kwargs']
		self.decode_strategy = decode_strategy_cls(self, infer_max_len, self.device, **decode_strategy_kwargs)

		self.to(self.device)

	def forward(self, src, trg, src_mask, trg_mask):
		# src = [batch size, src len]
		# trg = [batch size, trg len]

		e_output = self.encoder(src, src_mask)
		d_output, attn = self.decoder(trg, e_output, src_mask, trg_mask)
		output = self.out(d_output)

		return output, attn

	def load_checkpoint(self, model_dir, checkpoint=None, checkpoint_idx=0):
		"""
		Attempt to load past checkpoint into the model.
		If a specified checkpoint is set, load it;
		otherwise load the latest checkpoint in model_dir.
		Args:
			model_dir: location of the current model. Not used if checkpoint is specified
			checkpoint: location of the specific checkpoint to load
			checkpoint_idx: the epoch of the checkpoint
		NOTE: checkpoint_idx return -1 in the event of not found; while 0 is when checkpoint is forced
		"""
		if checkpoint is not None:
			saver.load_model(self, checkpoint)
			self._checkpoint_idx = checkpoint_idx
		else:
			if model_dir is not None:
				# load the latest available checkpoint, overriding the checkpoint value
				checkpoint_idx = saver.check_model(model_dir)
				if checkpoint_idx > 0:
					print(f'Found model with index {checkpoint_idx:d} already saved.')
					saver.load_model(self, model_dir, checkpoint_idx=checkpoint_idx)
				else:
					print('No checkpoint found, start from beginning.')
					checkpoint_idx = -1
			else:
				print('No model_dir available, start from beginning.')
				# train the model from begin
				checkpoint_idx = -1
			self._checkpoint_idx = checkpoint_idx

	def make_masks(self, src, trg):
		# src = [batch_size, src_len]
		# trg = [batch_size, trg_len]

		src_mask = (src != self.src_pad_idx)[:, None, None]
		# src_mask = [batch_size, 1, 1, src_len]

		trg_pad_mask = (trg != self.trg_pad_idx)[:, None, None]
		# trg_pad_mask = [batch_size, 1, 1, trg_len]
		trg_len = trg.shape[1]
		trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
		# trg_sub_mask = [trg_len, trg_len]
		trg_mask = trg_pad_mask & trg_sub_mask
		# trg_mask = [batch_size, 1, trg_len, trg_len]

		return src_mask, trg_mask

	def train_step(self, optimizer, batch, criterion):
		self.train()

		# move data to specific device's memory
		src = batch.src.transpose(0, 1).to(self.device)
		trg = batch.trg.transpose(0, 1).to(self.device)

		trg_input = trg[:, :-1]
		ys = trg[:, 1:].contiguous().view(-1)

		# create mask and perform network forward
		src_mask, trg_mask = self.make_masks(src, trg_input)
		preds = self(src, trg_input, src_mask, trg_mask)

		# perform backprogation
		optimizer.zero_grad()
		loss = criterion(preds.view(-1, preds.size(-1)), ys)
		loss.backward()
		optimizer.step_and_update_lr()
		loss = loss.item()

		return loss

	def validate(self, valid_iter, criterion, maximum_length=None):
		"""
		Compute loss in validation dataset.
		As we can't perform trimming the input in the valid_iter yet,
		using a crutch in maximum_input_length variable
		Args:
			valid_iter: the Iteration containing batches of data, accessed by .src and .trg
			criterion: the loss function to use to evaluate
			maximum_length: if fed, a tuple of max_input_len, max_output_len to trim the src/trg
		Returns:
			the avg loss of the criterion
		"""
		self.eval()

		with torch.no_grad():
			total_loss = []
			for batch in valid_iter:
				# load model into specific device (GPU/CPU) memory
				src = batch.src.transpose(0, 1).to(self.device)
				trg = batch.trg.transpose(0, 1).to(self.device)
				if maximum_length is not None:
					src = src[:, :maximum_length[0]]
					trg = trg[:, :maximum_length[1] - 1]  # using partials
				trg_input = trg[:, :-1]
				ys = trg[:, 1:].contiguous().view(-1)

				# create mask and perform network forward
				src_mask, trg_mask = self.make_masks(src, trg_input)
				preds = self(src, trg_input, src_mask, trg_mask)

				# compute loss on current batch
				loss = criterion(preds.view(-1, preds.size(-1)), ys)
				loss = loss.item()
				total_loss.append(loss)

		avg_loss = np.mean(total_loss)
		return avg_loss

	def run_train(self, model_dir):
		opt = self.config

		logging = init_logger(model_dir, opt['log_file_models'])

		trg_pad = self.TRG.vocab.stoi['<pad>']

		logging.info(f'{self.loader._lang_tuple[0]} * src vocab size = {len(self.SRC.vocab)}')
		logging.info(f'{self.loader._lang_tuple[1]} * trg vocab size = {len(self.TRG.vocab)}')
		logging.info('Building model...')

		model = self.to(self.device)

		checkpoint_idx = self._checkpoint_idx
		if checkpoint_idx < 0:
			# initialize weights
			print('Zero checkpoint detected, reinitialize the model')
			for p in model.parameters():
				if p.dim() > 1:
					nn.init.xavier_uniform_(p)
			checkpoint_idx = 0

		# also, load the scores of the best model
		best_model_score = saver.load_model_score(model_dir)

		# set up optimizer
		optim_algo = opt['optimizer']
		lr = opt['lr']
		d_model = opt['d_model']
		n_warmup_steps = opt['n_warmup_steps']
		optimizer_params = opt['optimizer_params']

		assert optim_algo in optimizers, f'Unknown optimizer: {optim_algo}'

		optimizer = ScheduledOptim(
			optimizer=optimizers[optim_algo](model.parameters(), **optimizer_params),
			init_lr=lr,
			d_model=d_model,
			n_warmup_steps=n_warmup_steps
		)

		# define loss function
		criterion = LabelSmoothingLoss(len(self.TRG.vocab), padding_idx=trg_pad, smoothing=opt['label_smoothing'])

		logging.info(self)
		model_encoder_parameters = filter(lambda p: p.requires_grad, self.encoder.parameters())
		model_decoder_parameters = filter(lambda p: p.requires_grad, self.decoder.parameters())
		params_encode = sum([np.prod(p.size()) for p in model_encoder_parameters])
		params_decode = sum([np.prod(p.size()) for p in model_decoder_parameters])

		logging.info(f'Encoder: {params_encode}')
		logging.info(f'Decoder: {params_decode}')
		logging.info(f'Number of parameters: {params_encode + params_decode}')
		logging.info(f'Starting training on {self.device}')
		for epoch in range(checkpoint_idx, opt['epochs']):
			total_loss = 0.0

			s = time.time()
			for i, batch in enumerate(self.train_iter):
				loss = self.train_step(optimizer, batch, criterion)
				total_loss += loss

				# print training loss after every {print_every} steps
				if (i + 1) % opt['print_every'] == 0:
					avg_loss = total_loss / opt['print_every']
					et = time.time() - s
					logging.info(
						f'epoch: {epoch:03d} - '
						f'iter: {i + 1:05d} - '
						f'train loss: {avg_loss:.4f} - '
						f'time elapsed/per batch: {et:.4f} {et / opt["print_every"]:.4f}'
					)
					total_loss = 0
					s = time.time()

				# bleu calculation and evaluate, save checkpoint for every {save_checkpoint_epochs} epochs
				s = time.time()
				valid_loss = self.validate(self.valid_iter,
				                           criterion,
				                           maximum_length=(self.encoder._max_seq_length, self.decoder._max_seq_length))

				if (epoch + 1) % opt['save_checkpoint_epochs'] == 0 and model_dir is not None:
					# valid_src_lang, valid_trg_lang = self.loader.lang_tuple
					bleuscore = bleu_batch_iter(self, self.valid_iter)

					saver.save_and_clear_model(model, model_dir,
					                           checkpoint_idx=epoch + 1,
					                           maximum_saved_model=opt['maximum_saved_model_train'])
					# keep the best model per every bleu calculation
					best_model_score = saver.save_best_model(model,
					                                         model_dir,
					                                         best_model_score,
					                                         bleuscore,
					                                         maximum_saved_model=opt['maximum_saved_model_eval'])
					logging.info(
						f'epoch: {epoch:03d} - '
						f'iter: {i:05d} - '
						f'valid loss: {valid_loss:.4f} - '
						f'bleu score: {bleuscore:.4f} - '
						f'full evaluation time: {time.time() - s:.4f}'
					)

				else:
					logging.info(
						f'epoch: {epoch:03d} - '
						f'iter: {i:05d} - '
						f'valid loss: {valid_loss:.4f} - '
						f'validation time: {time.time() - s:.4f}')

	def transl_batch(self, batch: list[str], input_max_length):
		"""Translate a single batch of sentences. Split to aid serving"""
		translated_batch = self.decode_strategy.transl_batch(batch, src_size_limit=input_max_length, output_tokens=True)
		return self.loader.detokenize(translated_batch)

	def transl_sentences(self, sentences: list[str], batch_size):
		"""Translate sentences by splitting them to batches and process them simultaneously"""
		self.eval()

		input_max_length = self.config['input_max_length']

		batches = [sentences[i: i + batch_size] for i in range(0, len(sentences), batch_size)]
		return [self.transl_batch(batch, input_max_length) for batch in batches]

	def run_infer(self, features_file, predictions_file):
		# load model into specific device (GPU/CPU) memory
		self.to(self.device)

		# Read inference file
		print(f'Reading features file from {features_file}...')
		with io.open(features_file, 'r', encoding='utf-8') as file:
			inputs = [line.strip() for line in file.readlines()]
		batch_size = self.config['valid_batch_size']

		print('Performing inference ...')
		# Translate by batched versions
		start = time.time()
		results = '\n'.join(self.transl_sentences(inputs, batch_size))
		print(f'Inference done, cost {time.time() - start:.2f} secs.')

		# Write results to system file
		print(f'Writing results to {predictions_file} ...')
		with io.open(predictions_file, 'w', encoding='utf-8') as file:
			file.write(results)

		print('All done!')

	def encode(self, *args, **kwargs):
		return self.encoder(*args, **kwargs)

	def decode(self, *args, **kwargs):
		return self.decoder(*args, **kwargs)

	# function to include the logits.
	# TODO use this in inference fns as well
	def to_logits(self, inputs):
		return self.out(inputs)
