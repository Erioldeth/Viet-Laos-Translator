from nltk.corpus import wordnet


# get_synonym  replace word with any synonym found among src
def get_synonym(word, SRC):
	syns = wordnet.synsets(word)
	for s in syns:
		for l in s.lemmas():
			if SRC.vocab.stoi[l.name()] != 0:
				return SRC.vocab.stoi[l.name()]

	return 0
