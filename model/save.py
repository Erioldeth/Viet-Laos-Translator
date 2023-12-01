import io
import os

import dill
import torch

EXTENSION = '.pkl'
MODEL_FILE_FORMAT = '{:s}{:s}'
VOCAB_FILE_FORMAT = '{:s}{:s}{:s}'


def make_vocab_path(path, lang_tuple):
	src_ext, trg_ext = lang_tuple
	src_vocab_path = os.path.join(path, VOCAB_FILE_FORMAT.format('vocab', src_ext, EXTENSION))
	trg_vocab_path = os.path.join(path, VOCAB_FILE_FORMAT.format('vocab', trg_ext, EXTENSION))
	return src_vocab_path, trg_vocab_path


def is_files(*paths):
	return all([os.path.isfile(path) for path in paths])


def save_vocab(path, lang_tuple, fields, force_new=False):
	src_field, trg_field = fields
	src_vocab_path, trg_vocab_path = make_vocab_path(path, lang_tuple)

	if not force_new and is_files(src_vocab_path, trg_vocab_path):
		return

	with (io.open(src_vocab_path, 'wb') as src_vocab_file,
	      io.open(trg_vocab_path, 'wb') as trg_vocab_file):
		dill.dump(src_field.vocab, src_vocab_file)
		dill.dump(trg_field.vocab, trg_vocab_file)


def load_vocab(path, lang_tuple, fields):
	src_field, trg_field = fields
	src_vocab_path, trg_vocab_path = make_vocab_path(path, lang_tuple)

	if not is_files(src_vocab_path, trg_vocab_path):
		return False

	with (io.open(src_vocab_path, 'rb') as src_vocab_file,
	      io.open(trg_vocab_path, 'rb') as trg_vocab_file):
		src_field.vocab = dill.load(src_vocab_file)
		trg_field.vocab = dill.load(trg_vocab_file)

	return True


def save_model(model, path):
	save_path = os.path.join(path, MODEL_FILE_FORMAT.format('model', EXTENSION))
	torch.save(model.state_dict(), save_path)
