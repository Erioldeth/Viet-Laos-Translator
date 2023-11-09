import io
import json
import os
import re
from shutil import copy2 as copy

import dill as pickle
import torch

MODEL_EXTENSION = '.pkl'
MODEL_FILE_FORMAT = '{:s}_{:d}{:s}'
BEST_MODEL_FILE = '.model_score.txt'
MODEL_SERVE_FILE = '.serve.txt'
VOCAB_FILE_FORMAT = '{:s}{:s}{:s}'


def save_vocab(path, language_tuple, fields, name_prefix='vocab', check_saved_vocab=True):
	src_field, trg_field = fields
	src_ext, trg_ext = language_tuple
	src_vocab_path = os.path.join(path, VOCAB_FILE_FORMAT.format(name_prefix, src_ext, MODEL_EXTENSION))
	trg_vocab_path = os.path.join(path, VOCAB_FILE_FORMAT.format(name_prefix, trg_ext, MODEL_EXTENSION))
	# do nothing if already exist
	if check_saved_vocab and os.path.isfile(src_vocab_path) and os.path.isfile(trg_vocab_path):
		return
	with io.open(src_vocab_path, 'wb') as src_vocab_file:
		pickle.dump(src_field.vocab, src_vocab_file)
	with io.open(trg_vocab_path, 'wb') as trg_vocab_file:
		pickle.dump(trg_field.vocab, trg_vocab_file)


def load_vocab(path, lang_tuple, fields, name_prefix='vocab'):
	"""
	Load the vocabulary from path into respective fields.
	If files doesn't exist, return False; if loaded properly, return True
	"""
	src_field, trg_field = fields
	src_ext, trg_ext = lang_tuple
	src_vocab_file_path = os.path.join(path, VOCAB_FILE_FORMAT.format(name_prefix, src_ext, MODEL_EXTENSION))
	trg_vocab_file_path = os.path.join(path, VOCAB_FILE_FORMAT.format(name_prefix, trg_ext, MODEL_EXTENSION))
	if not os.path.isfile(src_vocab_file_path) or not os.path.isfile(trg_vocab_file_path):
		# the vocab file wasn't dumped, return False
		return False
	with (io.open(src_vocab_file_path, 'rb') as src_vocab_file,
	      io.open(trg_vocab_file_path, 'rb') as trg_vocab_file):
		src_vocab = pickle.load(src_vocab_file)
		src_field.vocab = src_vocab
		trg_vocab = pickle.load(trg_vocab_file)
		trg_field.vocab = trg_vocab
	return True


def save_model(model, path, name_prefix='model', checkpoint_idx=0, save=True):
	save_path = os.path.join(path, MODEL_FILE_FORMAT.format(name_prefix, checkpoint_idx, MODEL_EXTENSION))
	torch.save(model.state_dict(), save_path)
	if save:
		save_vocab(path, model.loader._lang_tuple, model.fields)


def load_model(model, path, name_prefix='model', checkpoint_idx=0):
	if os.path.isdir(path):
		path = os.path.join(path, MODEL_FILE_FORMAT.format(name_prefix, checkpoint_idx, MODEL_EXTENSION))
	model.load_state_dict(torch.load(path))


def check_model(path, name_prefix='model', get_all_checkpoint=False):
	if not os.path.isdir(path):
		return 0

	model_re = re.compile(r'{:s}_(\d+){:s}'.format(name_prefix, MODEL_EXTENSION))
	matches = [re.match(model_re, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

	indices = sorted([int(m.group(1)) for m in matches if m is not None])
	return indices if get_all_checkpoint else indices[-1] if indices else 0


def save_and_clear_model(model, path, name_prefix='model', checkpoint_idx=0, maximum_saved_model=5):
	"""
	Keep only last n model when saving.
	Explicitly save the model regardless of its checkpoint index,
	e.g. if checkpoint_idx=0 & model 3 4 5 6 7 is in path, it will remove 3 and save 0 instead.
	"""
	indices = check_model(path, name_prefix=name_prefix, get_all_checkpoint=True)
	if maximum_saved_model <= len(indices):
		# remove model until n-1 model are left
		for i in indices[:-(maximum_saved_model - 1)]:
			os.remove(os.path.join(path, MODEL_FILE_FORMAT.format(name_prefix, i, MODEL_EXTENSION)))
	# perform save as normal
	save_model(model, path, name_prefix=name_prefix, checkpoint_idx=checkpoint_idx)


def load_model_score(path, score_file=BEST_MODEL_FILE):
	"""Load the model score as a list from a json dump, organized from best to worst."""
	score_file_path = os.path.join(path, score_file)
	if not os.path.isfile(score_file_path):
		return []
	with io.open(score_file_path, 'r') as jf:
		return json.load(jf)


def write_model_score(path, score_obj, score_file=BEST_MODEL_FILE):
	with io.open(os.path.join(path, score_file), 'w') as jf:
		json.dump(score_obj, jf)


def save_best_model(model, path, score_obj, model_metric, best_model_prefix='best_model', maximum_saved_model=5,
                    score_file=BEST_MODEL_FILE, save_after_update=True):
	worst_score = score_obj[-1] if len(score_obj) > 0 else -1.0
	if model_metric > worst_score:
		# perform update, overriding a slot or create new if needed
		insert_loc = next((idx for idx, score in enumerate(score_obj) if model_metric > score), 0)
		# every model below it, up to {maximum_saved_model}, will be moved down an index
		for i in range(insert_loc, min(len(score_obj), maximum_saved_model) - 1):
			# -1, due to the model are copied up to +1
			old_loc = os.path.join(path, MODEL_FILE_FORMAT.format(best_model_prefix, i, MODEL_EXTENSION))
			new_loc = os.path.join(path, MODEL_FILE_FORMAT.format(best_model_prefix, i + 1, MODEL_EXTENSION))
			copy(old_loc, new_loc)
		# save the model to the selected loc
		save_model(model, path, name_prefix=best_model_prefix, checkpoint_idx=insert_loc)
		# update the score obj
		score_obj.insert(insert_loc, model_metric)
		score_obj = score_obj[:maximum_saved_model]
		# also update in disk, if enabled
		if save_after_update:
			write_model_score(path, score_obj, score_file=score_file)
	# after routine had been done, return the obj
	return score_obj
