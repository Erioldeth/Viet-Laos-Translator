import io
import time

import numpy as np
import torch
import torch.nn as nn

import utils.save as saver
from config import *
from module.inference import strategies
from module.layer import *
from module.loader import Loader
from module.optim import optimizers, ScheduledOptim
from utils.decode_old import create_masks, translate_sentence
from utils.loss import LabelSmoothingLoss
from utils.metric import bleu_batch_iter


class Transformer(nn.Module):
	"""
	Implementation of Transformer architecture based on the paper `Attention is all you need`.
	Source: https://arxiv.org/abs/1706.03762
	"""

	def __init__(self, mode, model_dir, config):
		super().__init__()

		assert mode in ['train', 'infer'], f'Unknown mode: {mode}'
		assert model_dir is not None and config is not None, 'Missing model_dir and config'

		opt = self.config = Config(config)
		self.device = opt.get('device', const.DEFAULT_DEVICE)

		assert 'data' in opt, 'No data in config'
		assert 'train_data_location' in opt['data'], 'No training data in config'
		assert 'eval_data_location' in opt['data'], 'No evaluation data in config'

		data_opt = opt['data']
		self.loader = Loader(train_path=data_opt['train_data_location'],
		                     eval_path=data_opt['eval_data_location'],
		                     lang_tuple=(data_opt['src_lang'], data_opt['trg_lang']),
		                     option=opt)

		# input fields
		# TODO: improve this step
		self.SRC, self.TRG = self.loader.build_field(lower=opt.get('lowercase', const.DEFAULT_LOWERCASE))

		# initialize dataset and by proxy the vocabulary
		match mode:
			case 'train':
				# training flow, necessitate the DataLoader and iterations.
				# This will attempt to load vocab file from the dir instead of rebuilding,
				# but can build a new vocab if no data is found
				self.train_iter, self.valid_iter = self.loader.create_iterator(self.fields, model_dir)
			case 'infer':
				# inference, require model and vocab in the path
				self.loader.build_vocab(self.fields, model_dir)

		# define the model
		src_vocab_size, trg_vocab_size = len(self.SRC.vocab), len(self.TRG.vocab)
		d_model, N, heads, dropout = opt['d_model'], opt['n_layers'], opt['heads'], opt['dropout']
		# get the maximum amount of tokens per sample in encoder.
		# This is useful due to PositionalEncoder requiring this value
		train_ignore_length = self.config.get('train_max_length', const.DEFAULT_TRAIN_MAX_LENGTH)
		input_max_length = self.config.get('input_max_length', const.DEFAULT_INPUT_MAX_LENGTH)
		infer_max_length = self.config.get('max_length', const.DEFAULT_MAX_LENGTH)
		encoder_max_length = max(input_max_length, train_ignore_length)
		decoder_max_length = max(infer_max_length, train_ignore_length)
		self.encoder = Encoder(src_vocab_size, d_model, N, heads, dropout, max_seq_length=encoder_max_length)
		self.decoder = Decoder(trg_vocab_size, d_model, N, heads, dropout, max_seq_length=decoder_max_length)
		self.out = nn.Linear(d_model, trg_vocab_size)

		# load the beamsearch obj with preset values read from config.
		# ALWAYS require the current model, max_length, and device used as per DecodeStrategy base
		decode_strategy_cls = strategies[opt.get('decode_strategy', const.DEFAULT_DECODE_STRATEGY)]
		decode_strategy_kwargs = opt.get('decode_strategy_kwargs', const.DEFAULT_STRATEGY_KWARGS)
		self.decode_strategy = decode_strategy_cls(self, infer_max_length, self.device, **decode_strategy_kwargs)

		self.to(self.device)

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

	def forward(self, src, trg, src_mask, trg_mask, output_attention=False):
		"""Run a full model with specified source-target batched set of data
		Args:
			src: the source input [batch_size, src_len]
			trg: the target input (& expected output) [batch_size, trg len]
			src_mask: the padding mask for src [batch_size, 1, src_len]
			trg_mask: the triangle mask for trg [batch_size, trg_len, trg_len]
			output_attention: if specified, output the attention as needed
		Returns:
			the logits (unsoftmaxed outputs), same shape as trg
		"""
		e_outputs = self.encoder(src, src_mask)
		d_output, attn = self.decoder(trg, e_outputs, src_mask, trg_mask, output_attention=True)
		output = self.out(d_output)

		return output, attn if output_attention else output

	def train_step(self, optimizer, batch, criterion):
		"""
		Perform one training step.
		"""
		self.train()

		# move data to specific device's memory
		src = batch.src.transpose(0, 1).to(self.device)
		trg = batch.trg.transpose(0, 1).to(self.device)

		trg_input = trg[:, :-1]
		src_pad = self.SRC.vocab.stoi['<pad>']
		trg_pad = self.TRG.vocab.stoi['<pad>']
		ys = trg[:, 1:].contiguous().view(-1)

		# create mask and perform network forward
		src_mask, trg_mask = create_masks(src, trg_input, src_pad, trg_pad, self.device)
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

		src_pad = self.SRC.vocab.stoi['<pad>']
		trg_pad = self.TRG.vocab.stoi['<pad>']

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
				src_mask, trg_mask = create_masks(src, trg_input, src_pad, trg_pad, self.device)
				preds = self(src, trg_input, src_mask, trg_mask)

				# compute loss on current batch
				loss = criterion(preds.view(-1, preds.size(-1)), ys)
				loss = loss.item()
				total_loss.append(loss)

		avg_loss = np.mean(total_loss)
		return avg_loss

	def translate_sentence(self, sentence, device=None, k=None, max_len=None, debug=False):
		"""
		Receive a sentence string and output the prediction generated from the model.
		NOTE: sentence input is a list of tokens instead of string due to change in loader. See the current DefaultLoader for further details
		"""
		self.eval()
		if device is None:
			device = self.device
		if k is None:
			k = self.config.get('k', const.DEFAULT_K)
		if max_len is None:
			max_len = self.config.get('max_length', const.DEFAULT_MAX_LENGTH)

		# Get output from decode
		translated_tokens = translate_sentence(sentence, self, self.SRC, self.TRG, device, k, max_len, debug=debug,
		                                       output_list_of_tokens=True)

		return translated_tokens

	def translate_batch_sentence(self, sentences, trg_lang=None, output_tokens=False, batch_size=None):
		"""Translate sentences by splitting them to batches and process them simultaneously
		Args:
			sentences: the sentences in a list. Must NOT have been tokenized (due to SRC preprocess)
			output_tokens: if set, do not detokenize the output
			batch_size: if specified, use the value; else use config ones
		Returns:
			a matching translated sentences list in [detokenized format using loader.detokenize | list of tokens]
		"""
		if batch_size is None:
			batch_size = self.config.get('eval_batch_size', const.DEFAULT_EVAL_BATCH_SIZE)
		input_max_length = self.config.get('input_max_length', const.DEFAULT_INPUT_MAX_LENGTH)
		self.eval()

		translated = []
		for b_idx in range(0, len(sentences), batch_size):
			batch = sentences[b_idx: b_idx + batch_size]

			trans_batch = self.translate_batch(batch, trg_lang=trg_lang, output_tokens=output_tokens,
			                                   input_max_length=input_max_length)

			translated.extend(trans_batch)
		# for line in trans_batch:
		# 	print(line)
		return translated

	def translate_batch(self, batch_sentences, trg_lang=None, output_tokens=False, input_max_length=None):
		"""Translate a single batch of sentences. Split to aid serving
		Args:
			sentences: the sentences in a list. Must NOT have been tokenized (due to SRC preprocess)
			src_lang/trg_lang: the language from src->trg. Used for multilingual model only.
			output_tokens: if set, do not detokenize the output
		Returns:
			a matching translated sentences list in [detokenized format using loader.detokenize | list of tokens]
		"""
		if input_max_length is None:
			input_max_length = self.config.get('input_max_length', const.DEFAULT_INPUT_MAX_LENGTH)
		translated_batch = self.decode_strategy.translate_batch(batch_sentences,
		                                                        trg_lang=trg_lang,
		                                                        src_size_limit=input_max_length,
		                                                        output_tokens=True,
		                                                        debug=False)
		return self.loader.detokenize(translated_batch) if not output_tokens else translated_batch

	def run_train(self, model_dir=None):
		opt = self.config
		from utils.logger import init_logger
		logging = init_logger(model_dir, opt.get('log_file_models'))

		trg_pad = self.TRG.vocab.stoi['<pad>']
		# load model into specific device (GPU/CPU) memory
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
		optimizer_params = opt.get('optimizer_params', {})

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
					valid_src_lang, valid_trg_lang = self.loader.lang_tuple
					bleuscore = bleu_batch_iter(self, self.valid_iter, src_lang=valid_src_lang, trg_lang=valid_trg_lang)

					saver.save_and_clear_model(model,
					                           model_dir,
					                           checkpoint_idx=epoch + 1,
					                           maximum_saved_model=opt.get('maximum_saved_model_train',
					                                                       const.DEFAULT_NUM_KEEP_MODEL_TRAIN))
					# keep the best model per every bleu calculation
					best_model_score = saver.save_best_model(model,
					                                         model_dir,
					                                         best_model_score,
					                                         bleuscore,
					                                         maximum_saved_model=opt.get('maximum_saved_model_eval',
					                                                                     const.DEFAULT_NUM_KEEP_MODEL_TRAIN))
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

	def run_infer(self, features_file, predictions_file, src_lang=None, trg_lang=None, batch_size=None):
		opt = self.config
		# load model into specific device (GPU/CPU) memory
		model = self.to(self.device)

		# Read inference file
		print('Reading features file from {}...'.format(features_file))
		with io.open(features_file, 'r', encoding='utf-8') as read_file:
			inputs = [l.strip() for l in read_file.readlines()]

		print('Performing inference ...')
		# Append each translated sentence line by line
		#        results = '\n'.join([model.loader.detokenize(model.translate_sentence(sentence)) for sentence in inputs])
		# Translate by batched versions
		start = time.time()
		results = '\n'.join(
			self.translate_batch_sentence(inputs, src_lang=src_lang, trg_lang=trg_lang, output_tokens=False,
			                              batch_size=batch_size))
		print(f'Inference done, cost {time.time() - start:.2f} secs.')

		# Write results to system file
		print(f'Writing results to {predictions_file} ...')
		with io.open(predictions_file, 'w', encoding='utf-8') as write_file:
			write_file.write(results)

		print('All done!')

	def encode(self, *args, **kwargs):
		return self.encoder(*args, **kwargs)

	def decode(self, *args, **kwargs):
		return self.decoder(*args, **kwargs)

	# function to include the logits.
	# TODO use this in inference fns as well
	def to_logits(self, inputs):
		return self.out(inputs)

	@property
	def fields(self):
		return self.SRC, self.TRG
