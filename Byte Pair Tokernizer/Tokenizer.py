import argparse
import os
from nltk import wordpunct_tokenize, sent_tokenize
import re
import json


def get_bpe_vocab(path):
    with open("data.txt", 'r') as file:  # Reading the text file
        data = file.read()
    vocab = dict()
    data = data.lower()
    sentences = sent_tokenize(data)  # Creating the list of sentences from the given text corpus
    for sentence in sentences:
        words = wordpunct_tokenize(sentence)  # Creating the list of words present in sentence
        for word in words:
            vocab[word] = vocab.get(word, 0) + 1  # Increasing the frequency of word by one
    bpe_vocab = dict()
    for token in vocab:
        ntoken = ' '.join(list(
            token)) + ' </w>'  # Splitting the word and joining it with spaces in between and adding a end of word token
        bpe_vocab[ntoken] = vocab[token]
    return bpe_vocab


def get_pair_counts(vocab):  # Functions returns the pair count of each unique pair
    pairs = dict()
    for word in vocab:
        contents = word.split(' ')
        for u, v in zip(contents[:-1], contents[1:]):
            if (u, v) not in pairs:
                pairs[(u, v)] = 0
            pairs[(u, v)] += vocab[word]
    return pairs


def merge_vocab(pair, vocab):
    bigram = ' '.join(list(pair))  # Creating a bigram by inserting a space between the pair and joining them
    new_vocab = dict()
    p = re.compile(
        r'(?<!\S)' + bigram + r'(?!\S)')  # Regular Expression so that the bigram is between the two white spaces
    for word in vocab:
        w_out = p.sub(''.join(pair), word)  # Forming a new word by replacing the Regex pattern with a single token
        new_vocab[w_out] = vocab[word]
    return new_vocab


def encode_decode(vocab):  # Create a mapping from token to int and int to token
    vocab_to_idx = {}
    idx_to_vocab = {}
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)  # Sorting the vocab by count of the words
    for i in range(len(sorted_vocab)):
        word, count = sorted_vocab[i]
        vocab_to_idx[word] = i
        idx_to_vocab[i] = word
    return vocab_to_idx, idx_to_vocab


def count_token_freq(vocab):  # Create a dictionary that contains the count of each token
    freq = {}
    for word in vocab:  # Iterating through the vocab
        tokens = word.split()  # Creating the list of the tokens in a particular word
        for token in tokens:  # Iterating through each token and increasing its frequency
            if token not in freq:
                freq[token] = 0
            freq[token] += vocab[word]
    return freq


def train_tokenizer(vocab,
                    merges_count):  # This function performs merges_count number of merges on vocab and return the
    # mapping from token to int and int to token
    for i in range(merges_count):
        pairs = get_pair_counts(vocab)  # Dictionary that contains the frequency of each pair
        max_count_pair = max(pairs, key=pairs.get)  # Get the pair that occurs most frequently
        vocab = merge_vocab(max_count_pair, vocab)  # perform merging of the max occurring pair in the entire vocab
    freq = count_token_freq(vocab)  # Getting the frequency of each token
    freq["<unk>"] = 1  # Adding the token for unknown characters
    vocab_to_idx, idx_to_vocab = encode_decode(freq)  # Getting mapping
    return vocab_to_idx, idx_to_vocab, freq


def train(path_text_corpus, path_tokenizer, merges_count):
    bpe_vocab = get_bpe_vocab(path_text_corpus)
    vocab_to_idx, idx_to_vocab, freqs = train_tokenizer(bpe_vocab, merges_count)
    tokenizer_object = BytePairTokenizer(freqs, vocab_to_idx, idx_to_vocab)
    tokenizer_object.save(path_tokenizer)


class BytePairTokenizer:

    def __init__(self, freqs=None, vocab_to_idx=None, idx_to_vocab=None):
        self.freqs = freqs if freqs is not None else {}
        self.vocab_to_idx = vocab_to_idx if vocab_to_idx is not None else {}
        self.idx_to_vocab = idx_to_vocab if idx_to_vocab is not None else {}
        self.unk = "<unk>"
        self.eow = "</w>"

    def vocab_size(self):
        return len(self.vocab_to_idx)

    def get_token(self, idx):
        return self.idx_to_vocab[idx]

    def get_id(self, token):
        unk_id = self.vocab_to_idx[self.unk]
        idx = self.vocab_to_idx[token] if token in self.vocab_to_idx else unk_id
        return idx

    def get_tokens(self, idxs):
        tokens = []
        for idx in idxs:
            tokens.append(self.idx_to_vocab[idx])
        return tokens

    def get_ids(self, tokens):
        ids = []
        for token in tokens:
            if token in self.vocab_to_idx:
                ids.append(self.vocab_to_idx[token])
            else:
                ids.append(self.vocab_to_idx[self.unk])
        return ids

    def get_max_pair(self, tokens):  # If merge is possible in text then it returns the two indices
        pairs = {}
        for i in range(1, len(tokens)):
            pair = ''.join(tokens[i - 1:i + 1])
            if pair in self.freqs:
                pairs[(i - 1, i)] = self.freqs[pair]
        return None if len(pairs) == 0 else max(pairs, key=pairs.get)

    def merge_max_pair(self, tokens):  # Perform Merging
        max_pair = self.get_max_pair(tokens)
        merged = True if max_pair is not None else False
        if merged:
            tokens = tokens[:max_pair[0]] + [''.join(tokens[max_pair[0]:max_pair[1] + 1])] + tokens[max_pair[1] + 1:]
        return tokens, merged

    def merge_tokens(self, tokens):  # Performing merging till any merging is possible
        tokens, merged = self.merge_max_pair(tokens)
        while merged:
            tokens, merged = self.merge_max_pair(tokens)
        return tokens

    def encode(self, input):
        words = input.split(" ")
        tokens = []
        for word in words:
            word = list(word) + [self.eow]
            tokens += self.merge_tokens(word)
        enc = self.get_ids(tokens)
        return enc

    def decode(self, input):
        text = []
        word = ""
        tokens = self.get_tokens(input)
        for i in range(len(tokens)):
            word += tokens[i]
            if word.endswith("</w>"):
                text.append(word[:-4])
                word = ""
        return ' '.join(text)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f"{path}/freqs.json", 'w', encoding='utf-8') as outfile:
            json.dump(self.freqs, outfile, indent=4, ensure_ascii=False)

        with open(f"{path}/vocab_to_idx.json", 'w', encoding='utf-8') as outfile:
            json.dump(self.vocab_to_idx, outfile, indent=4, ensure_ascii=False)

        with open(f"{path}/idx_to_vocab.json", 'w', encoding='utf-8') as outfile:
            json.dump(self.idx_to_vocab, outfile, indent=4, ensure_ascii=False)

    def load(self, path):
        with open(f"{path}/freqs.json", 'r', encoding='utf-8') as file:
            freqs = json.load(file)

        with open(f'{path}/vocab_to_idx.json', 'r', encoding='utf-8') as file:
            vocab_to_idx = json.load(file)

        with open(f'{path}/idx_to_vocab.json', 'r', encoding='utf-8') as file:
            idx_to_vocab = json.load(file)

        idx_to_vocab = {int(k): v for k, v in
                        idx_to_vocab.items()}  # By default, json stores keys and values both as strings so we have
        # convert back in to int

        self.freqs = freqs
        self.vocab_to_idx = vocab_to_idx
        self.idx_to_vocab = idx_to_vocab


def main(path_text_corpus, path_tokenizer, merges_count):
    train(path_text_corpus, path_tokenizer, merges_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('path_text_corpus', type=str)
    parser.add_argument('path_tokenizer', type=str)
    parser.add_argument('merges_count', type=int)

    args = parser.parse_args()

    path_text_corpus = args.path_text_corpus
    path_tokenizer = args.path_tokenizer
    merges_count = args.merges_count

    main(path_text_corpus, path_tokenizer, merges_count)
