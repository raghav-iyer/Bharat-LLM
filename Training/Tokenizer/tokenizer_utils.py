import sentencepiece as spm

def load_tokenizer(model_prefix):
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp

def tokenize_text(tokenizer, text):
    return tokenizer.encode(text, out_type=int)

def detokenize_text(tokenizer, tokens):
    return tokenizer.decode(tokens)

# tokenizer = load_tokenizer("multilingual_tokenizer")
# tokenized_sentence = tokenize_text(tokenizer, "Hi")
# print(tokenized_sentence)
