Put trained model in this directory

tokenizer.vi.model {
    model_type=unigram
    vocab_size=20000
    max_sentence_length=100000
    split_by_whitespace=false
    input_sentence_size=700000
}
tokenizer.vi_bpe.model {
    model_type=bpe
    vocab_size=16000
    max_sentence_length=100000
    split_by_whitespace=false
    input_sentence_size=1000000
}