from torchtext.data import BucketIterator, Field
from torchtext.datasets import TranslationDataset

from config import const
from utils.save import load_vocab


class Loader:
	def __init__(self, train_path, eval_path=None, lang_tuple=None, option=None):
		"""Load training/eval data file pairing, process and create data iterator for training """
		self._train_path = train_path
		self._eval_path = eval_path
		self._lang_tuple = lang_tuple
		self._option = option

	def detokenize(self, tokens_list):
		"""Differentiate between [batch, len] and [len]; joining tokens back to strings"""
		if not tokens_list or isinstance(tokens_list[0], str):
			# [len], single sentence version
			return ' '.join(tokens_list)
		else:
			# [batch, len], batch sentence version
			return [' '.join(tokens) for tokens in tokens_list]

	def build_field(self, **kwargs):
		"""
		Build fields that will handle the conversion from token->idx and vice versa.
		TODO: improve (Lao is (sentence/phrase)-delimited | Vietnamese is syllable-delimited)
		"""
		return Field(**kwargs), Field(init_token=const.DEFAULT_SOS, eos_token=const.DEFAULT_EOS, **kwargs)

	def build_vocab(self, fields, model_path, data=None, **kwargs):
		"""
		Build the vocabulary object for torchtext Field.
		There are two flows:
		- if the model path is present, it will first try to load the pickled/dilled vocab object from path.
		This is accessed on continued training & standalone inference
		- if that failed and data is available, try to build the vocab from that data.
		This is accessed on first time training
		"""

		# the condition will try to load vocab pickled to model path.
		if not load_vocab(model_path, self._lang_tuple, fields):
			assert data is not None, 'No data to build vocab from'

			print('Building vocab from received data')
			# build the vocab using formatted data.
			src_field, trg_field = fields
			src_field.build_vocab(data, **kwargs)
			trg_field.build_vocab(data, **kwargs)
		else:
			print('Load vocab from path successful')

	def create_iterator(self, fields, model_path=None):
		"""
		Create the iterator needed to load batches of data and bind them to existing fields
		NOTE: unlike the previous loader, this one inputs list of tokens instead of a string,
		which necessitate redefining of translate_sentence pipe
		"""
		ext = self._lang_tuple
		token_limit = self._option.get('train_max_length', const.DEFAULT_TRAIN_MAX_LENGTH)
		filter_fn = lambda x: len(x.src) <= token_limit and len(x.trg) <= token_limit

		train_data = TranslationDataset(self._train_path, ext, fields, filter_pred=filter_fn)
		eval_data = TranslationDataset(self._eval_path, ext, fields)

		# now we can execute build_vocab.
		# This function will try to load vocab from model_path, and if failed, build the vocab from train_data
		build_vocab_kwargs = self._option.get('build_vocab_kwargs', {})
		self.build_vocab(fields, model_path, train_data, **build_vocab_kwargs)

		# crafting iterators
		train_iter = BucketIterator(train_data,
		                            batch_size=self._option.get('batch_size', const.DEFAULT_BATCH_SIZE),
		                            device=self._option.get('device', const.DEFAULT_DEVICE))

		eval_iter = BucketIterator(eval_data,
		                           batch_size=self._option.get('eval_batch_size', const.DEFAULT_EVAL_BATCH_SIZE),
		                           device=self._option.get('device', const.DEFAULT_DEVICE),
		                           train=False)

		return train_iter, eval_iter

	@property
	def lang_tuple(self):
		"""Loader will use the default lang option @bleu_batch_iter <sos>, hence, None"""
		return None, None