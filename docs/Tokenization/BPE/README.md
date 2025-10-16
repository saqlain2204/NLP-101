# Byte Pair Encoding (BPE)

**Basic Idea:** Train the tokenizer on raw text to automatically determine the vocabulary.

**Intuition:** Common sequences of characters are represented by a single token, rare sequences are represented by many tokens.

**Algorithm Sketch:**
1. Start with each character as a token (plus an end-of-word marker).
2. Count all adjacent token pairs in the corpus.
3. Merge the most frequent pair into a new token.
4. Repeat for a fixed number of merges.

## How It Works

BPE is an algorithm originally used for data compression. In NLP, it's used to build subword vocabularies:

1. **Initialization**: Start with all characters in your corpus as separate tokens
2. **Iterative Merging**: Repeatedly merge the most frequent adjacent pair of tokens
3. **Stopping Criterion**: Stop after a certain number of merges (vocabulary size)

## Step-by-Step BPE Process

Let's walk through a simplified example of how BPE works with these words:
```
low lower lowest
newer wider
```

### Initial Vocabulary
Each word is split into characters with an end-of-word marker:
- `l, o, w, </w>` (low)
- `l, o, w, e, r, </w>` (lower)
- `l, o, w, e, s, t, </w>` (lowest)
- `n, e, w, e, r, </w>` (newer)
- `w, i, d, e, r, </w>` (wider)

### First Few Merge Operations

1. **Find most frequent pair**: `e, r` (appears 3 times)
   - Merge: `e + r → er`
   - Vocabulary becomes: `l, o, w, </w>`, `l, o, w, er, </w>`, `l, o, w, e, s, t, </w>`, `n, e, w, er, </w>`, `w, i, d, er, </w>`

2. **Find most frequent pair**: `l, o` (appears 3 times)
   - Merge: `l + o → lo`
   - Vocabulary becomes: `lo, w, </w>`, `lo, w, er, </w>`, `lo, w, e, s, t, </w>`, `n, e, w, er, </w>`, `w, i, d, er, </w>`

3. **Find most frequent pair**: `lo, w` (appears 3 times)
   - Merge: `lo + w → low`
   - Vocabulary becomes: `low, </w>`, `low, er, </w>`, `low, e, s, t, </w>`, `n, e, w, er, </w>`, `w, i, d, er, </w>`

...and so on.

### Visual Progression for "lowest"

```
Initial:  l   o   w   e   s   t   </w>
Merge 1:  l   o   w   e   s   t   </w>  (er doesn't appear)
Merge 2:  lo  w   e   s   t   </w>      (lo created)
Merge 3:  low e   s   t   </w>          (low created)
...
Final:    low est </w>                   (after additional merges)
```

This demonstrates how the algorithm gradually builds larger subword units based on frequency.

## Implementation

The Python implementation includes several key functions that work together to perform Byte Pair Encoding:

### 1. Statistics Collection Function

```python
def get_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs
```

This function counts the frequency of adjacent token pairs in the vocabulary. For each word in our vocabulary:
- It iterates through all adjacent pairs of symbols
- It increments a counter for each pair, weighted by how frequently the word appears
- The result is a Counter object mapping each pair to its frequency across the corpus

### 2. Vocabulary Merging Function

```python
def merge_vocab(pair, vocab):
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
```

This function applies a specific merge operation to the entire vocabulary:
- It creates a new bigram by joining the pair of tokens
- For each word in the vocabulary, it replaces all occurrences of the specific pair with the new merged token
- It preserves word frequencies in the new vocabulary
- It returns a new vocabulary with the merge applied throughout

### 3. BPE Training Function

```python
def bpe(corpus, num_merges):
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
```

The main BPE algorithm function that:
- Initializes a vocabulary where each word is split into individual characters plus an end-of-word marker `</w>`
- Repeatedly:
  - Counts pair frequencies with `get_stats()`
  - Finds the most frequent pair
  - Applies the merge operation with `merge_vocab()`
  - Records the merge for later use in encoding
- Returns both the sequence of merges and the final vocabulary

### 4. Encoding Function

```python
def encode(word, merges):
    word = tuple(word) + ('</w>',)
    for pair in merges:
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
    # Handle the end-of-word marker
    if word[-1] == '</w>':
        return word[:-1]
    else:
        last = word[-1].replace('</w>', '')
        return word[:-1] + (last,) if last else word[:-1]
```

This function tokenizes new words using the learned BPE merges:
- It starts with the word split into individual characters plus the end marker
- It applies each merge operation in the same order they were learned during training
- It handles the end-of-word marker properly when returning the final tokenized word

## Usage Example

Here's a complete example showing how to use the BPE implementation:

```python
from bpe import bpe, encode

# Define a small corpus for training
corpus = [
    "low lower lowest",
    "newer wider",
]

# Train the BPE model with 20 merge operations
merges, vocab = bpe(corpus, num_merges=20)

# Print the sequence of merges learned
print("Learned merges:")
for i, merge in enumerate(merges):
    print(f"Merge {i+1}: {merge[0]} + {merge[1]} → {''.join(merge)}")

# Print the final vocabulary
print("\nFinal vocabulary:")
for word, freq in vocab.items():
    print(f"{word}: {freq}")

# Encode a new word using the learned merges
encoded = encode("lowest", merges)
print("\nEncoded 'lowest':", encoded)
```

This example demonstrates:
1. **Training**: Creating a BPE model from a small corpus
2. **Inspection**: Viewing the learned merges and resulting vocabulary
3. **Application**: Using the model to encode a new word

When you run this code, you'll see how the word "lowest" gets tokenized according to the subword units learned during training.

## Advantages of BPE

- Handles out-of-vocabulary words gracefully
- Balances word-level and character-level representations
- Efficient for morphologically rich languages
- Widely used in modern NLP systems (GPT, BERT, etc.)

---

*Created by Saqlain.*