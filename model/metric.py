from torchtext.data.metrics import bleu_score


def _revert_trg(sent, eos):  # revert batching process on sentence
	try:
		endloc = sent.index(eos)
		return sent[1:endloc]
	except ValueError:
		return sent[1:]


def bleu_batch_iter(model, valid_iter):
	"""
	Perform batched translations; other metrics are the same.
	Note that the inputs/outputs had been preprocessed,
	but have [batch_size, length] shape as per BucketIterator
	"""
	translated_batched_pair = (
		(
			batch.trg,  # transpose due to timestep-first batches
			model.decode_strategy.transl_batch(
				batch.src,
				output_tokens=True,
				replace_unk=False,  # do not replace in this version
			)
		)
		for batch in valid_iter
	)

	flattened_pair = (([model.TRG.vocab.itos[i] for i in trg], pred)
	                  for batch_trg, batch_pred in translated_batched_pair
	                  for trg, pred in zip(batch_trg, batch_pred))
	flat_labels, predictions = [list(l) for l in zip(*flattened_pair)]
	# remove <sos> and <eos> also updim the trg for 3D requirements.
	labels = [[_revert_trg(l, '<eos>')] for l in flat_labels]
	return bleu_score(predictions, labels)