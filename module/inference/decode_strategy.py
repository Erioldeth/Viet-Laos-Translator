import torch
from nltk.corpus import wordnet
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence


class DecodeStrategy(object):
	"""
	Base, abstract class for generation strategies.
	Contain specific call to base model that use it
	"""

	def __init__(self, model, max_len, device):
		self.model = model
		self.max_len = max_len
		self.device = device

	@property
	def SRC(self):
		return self.model.SRC

	@property
	def TRG(self):
		return self.model.TRG

	def preprocess_batch(self, sentences, pad_token="<pad>"):
		"""Feed an unprocessed batch into the torchtext.Field of source.
		Args:
			sentences: [batch_size] of str
			pad_token: the pad token used to pad the sentences
		Returns:
			the sentences in Tensor format, padded with pad_value"""

		# tokenizing
		processed_sent = list(map(self.SRC.preprocess, sentences))
		# convert to tensors and indices
		tokenized_sent = [torch.LongTensor([self._token_to_index(t) for t in s]) for s in processed_sent]
		# padding sentences
		sentences = Variable(pad_sequence(tokenized_sent, True, padding_value=self.SRC.vocab.stoi[pad_token]))
		return sentences

	def _token_to_index(self, tok):
		"""
		Implementing get_synonym as default.
		Override if want to use default behavior (<unk> for unknown words, independent of wordnet)
		"""
		return self.SRC.vocab.stoi[tok] \
			if self.SRC.vocab.stoi[tok] != self.SRC.vocab.stoi['<eos>'] \
			else get_synonym(tok, self.SRC)


def get_synonym(word, SRC):
	syns = wordnet.synsets(word)
	for s in syns:
		for l in s.lemmas():
			if SRC.vocab.stoi[l.name()] != 0:
				return SRC.vocab.stoi[l.name()]

	return 0
