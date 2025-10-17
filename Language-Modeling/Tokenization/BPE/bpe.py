# BPE Implementation

from collections import Counter

def get_stats(vocab):
    """Count frequency of all symbol pairs in the vocab."""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """Merge all occurrences of the most frequent pair in the vocab."""
    new_vocab = {}
    bigram = ''.join(pair)
    for word in vocab:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(bigram)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = vocab[word]
    return new_vocab
    
def bpe(corpus, num_merges):
    """Train BPE on a corpus and return the merges and final vocabulary."""
    vocab = Counter()
    for line in corpus:
        for word in line.split():
            vocab[tuple(word) + ('</w>',)] += 1
    
    merges = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merges.append(best)
    return merges, vocab

def encode(word, merges):
    """Encode a word using learned BPE merges."""
    word = tuple(word) + ('</w>',)
    print(f"Start: {word}")
    for idx, pair in enumerate(merges):
        i = 0
        new_word = []
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(''.join(pair))
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = tuple(new_word)
        # print(f"After merge {idx+1} {pair}: {word}")
    if word[-1] == '</w>':
        return word[:-1]
    else:
        last = word[-1].replace('</w>', '')
        return word[:-1] + (last,) if last else word[:-1]

## Example Usage

if __name__ == "__main__":
    corpus = [
    "low lower lowest",
    "newer wider",
    ]
    merges, vocab = bpe(corpus, num_merges=20)

    encoded = encode("lowest", merges)
    print("Encoded 'lowest':", encoded)