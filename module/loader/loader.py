from torchtext.data import BucketIterator, Field
from torchtext.datasets import TranslationDataset

from model.save import load_vocab, save_vocab


class Loader:
	def __init__(self, train_path, valid_path, lang_tuple, option):
		self.train_path = train_path
		self.valid_path = valid_path
		self.lang_tuple = lang_tuple
		self.option = option

	def build_vocab(self, fields: tuple[Field, Field], model_dir, data=None, **kwargs):
		if not load_vocab(fields, model_dir, self.lang_tuple):
			assert data is not None, 'No data to build vocab from'

			print('Building vocab from received data')

			src_field, trg_field = fields
			src_field.build_vocab(data, **kwargs)
			trg_field.build_vocab(data, **kwargs)

			save_vocab(fields, model_dir, self.lang_tuple)
		else:
			print('Load vocab from path successful')

	def create_iterator(self, fields: tuple[Field, Field], model_dir, device) -> tuple[BucketIterator, BucketIterator]:
		opt = self.option
		ext = self.lang_tuple
		token_limit = self.option['train_max_length']

		print('Building dataset ...')
		filter_fn = lambda x: len(x.src) <= token_limit and len(x.trg) <= token_limit
		train_data = TranslationDataset(self.train_path, ext, fields, filter_pred=filter_fn)
		valid_data = TranslationDataset(self.valid_path, ext, fields)

		print('Building vocab from dataset ...')
		self.build_vocab(fields, model_dir, train_data, **opt['build_vocab_kwargs'])

		return BucketIterator.splits((train_data, valid_data), [opt['train_batch_size']] * 2, device=device)
