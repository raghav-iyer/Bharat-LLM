from collections import defaultdict

class BPE:
    def __init__(self, num_merges: int):
        self.num_merges = num_merges
        self.vocab = {}

    def get_vocab(self, text: str):
        for word in text.split():
            word = ' '.join(list(word)) + ' </w>'
            if word in self.vocab:
                self.vocab[word] += 1
            else:
                self.vocab[word] = 1

    def get_stats(self):
        pairs = defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair):
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in self.vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = self.vocab[word]
        self.vocab = new_vocab

    def fit(self, text: str):
        self.get_vocab(text)
        for _ in range(self.num_merges):
            pairs = self.get_stats()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merge_vocab(best_pair)

    def encode(self, text: str):
        tokens = []
        for word in text.split():
            word = ' '.join(list(word)) + ' </w>'
            if word in self.vocab:
                tokens.append(word)
            else:
                tokens.append("OOV: " + word)
        return tokens
