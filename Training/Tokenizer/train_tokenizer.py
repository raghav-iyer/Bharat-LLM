import sentencepiece as spm

def train_sentencepiece_multilingual(file_paths, model_prefix, vocab_size):
    spm.SentencePieceTrainer.train(
        input=','.join(file_paths),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,  # Special tokens
        model_type='bpe',
        user_defined_symbols=["[SOS]", "[EOS]"] 
    )