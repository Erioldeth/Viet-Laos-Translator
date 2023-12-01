from torchtext.data import BucketIterator, Field
from torchtext.datasets import TranslationDataset

from model.save import load_vocab, save_vocab


class Loader:
	def __init__(self, train_path, valid_path, lang_tuple, option):

		self.train_path = train_path
		self.valid_path = valid_path
		self.lang_tuple = lang_tuple
		self.option = option

	def detokenize(self, tokens_list):
		"""Differentiate between [batch, len] and [len]; joining tokens back to strings"""
		if not tokens_list or isinstance(tokens_list[0], str):
			# [len], single sentence version
			return ' '.join(tokens_list)
		else:
			# [batch, len], batch sentence version
			return [' '.join(tokens) for tokens in tokens_list]

	def build_vocab(self, fields: tuple[Field, Field], model_dir, data=None, **kwargs):
		if not load_vocab(model_dir, self.lang_tuple, fields):
			assert data is not None, 'No data to build vocab from'

			print('Building vocab from received data')

			src_field, trg_field = fields
			src_field.build_vocab(data, **kwargs)
			trg_field.build_vocab(data, **kwargs)

			save_vocab(model_dir, self.lang_tuple, fields)
		else:
			print('Load vocab from path successful')

	def create_iterator(self, fields, model_dir, device):
		ext = self.lang_tuple
		token_limit = self.option['train_max_length']
		filter_fn = lambda x: len(x.src) <= token_limit and len(x.trg) <= token_limit

		train_data = TranslationDataset(self.train_path, ext, fields, filter_pred=filter_fn)
		valid_data = TranslationDataset(self.valid_path, ext, fields)

		build_vocab_kwargs = self.option['build_vocab_kwargs']
		self.build_vocab(fields, model_dir, train_data, **build_vocab_kwargs)

		# crafting iterators
		train_iter = BucketIterator(train_data,
		                            batch_size=self.option['batch_size'],
		                            device=device)

		valid_iter = BucketIterator(valid_data,
		                            batch_size=self.option['valid_batch_size'],
		                            device=device,
		                            train=False)

		return train_iter, valid_iter
