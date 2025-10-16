## BPE

**Basic Idea**: Train the tokenizer on raw text to automatically determine the vocabulary.

**Intuition**: Common sequences of characters are represented by a single token, rare sequences are represented by many tokens.

**Sketch**: Start with each byte as a token and successively merge the most common adjacent tokens.