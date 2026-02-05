# Tokenization Deep Dive

The tokenizer is a completely separate, independent module from the LLM. It has its own training dataset of text (which could be different from that of the LLM), on which you train the vocabulary using the Byte Pair Encoding (BPE) algorithm. It then translates back and forth between raw text and sequences of tokens. The LLM later only ever sees the tokens and never directly deals with any text.

---

## PART 1: THE BIG PICTURE (5-Second Mental Model)

```
Text "Hello world" → UTF-8 Bytes → BPE Merges → Tokens [104, 101, 108, 108, 111, ...] → LLM Input
```

**The core idea**: Tokenization converts raw text into integer sequences that the neural network can process, while compressing the representation by learning common character sequences.

---

## PART 2: WHY TOKENIZATION MATTERS (The "Root of Suffering")

Tokenization is at the heart of much weirdness of LLMs. Do not brush it off.

- **Why can't LLM spell words?** Tokenization (word may be split across tokens)
- **Why can't LLM do super simple string processing tasks like reversing a string?** Tokenization (operations on tokens, not characters)
- **Why is LLM worse at non-English languages (e.g., Japanese)?** Tokenization (less efficient for non-Latin scripts)
- **Why is LLM bad at simple arithmetic?** Tokenization (numbers split unpredictably: "127", "677", "804")
- **Why did GPT-2 have more than necessary trouble coding in Python?** Tokenization (indentation, special characters)
- **Why did my LLM abruptly halt when it sees the string "<|endoftext|>"?** Tokenization (special token triggers early stop)
- **What is this weird warning I get about a "trailing whitespace"?** Tokenization (whitespace often separate token)
- **Why the LLM break if I ask it about "SolidGoldMagikarp"?** Tokenization (rare token, may not exist in vocab)
- **Why should I prefer to use YAML over JSON with LLMs?** Tokenization (YAML uses fewer/simpler tokens)
- **Why is LLM not actually end-to-end language modeling?** Tokenization (text → tokens → model → tokens → text, not direct text modeling)
- **What is the real root of suffering?** Tokenization.

**Concrete examples**:
```
127 + 677 = 804      → Tokens: [127, +, 677, =, 804]    → LLM struggles
1275 + 6773 = 8041    → Tokens: [1275, +, 6773, =, 8041]  → Different tokens!

Egg.                   → 3 tokens
I have an Egg.          → 5 tokens (capitalized "Egg" different!)
egg.                   → 3 tokens
EGG.                   → 3 tokens (case differences)

만나서 반가워요.          → Many tokens per character (Korean inefficient)
```

---

## PART 3: UTF-8 ENCODING BASICS

### How Text Becomes Bytes

**Option 1: Using ord()**
```python
[ord(x) for x in "hello, this is an LLM"]
# Output: [104, 101, 108, 108, 111, 44, 32, 116, 104, 105, 115, 32, 105, 115, 32, 97, 110, 32, 76, 76, 77]
```

**Option 2: Using UTF-8 encode (equivalent)**
```python
list("hello, this is an LLM".encode("utf-8"))
# Output: [104, 101, 108, 108, 111, 44, 32, 116, 104, 105, 115, 32, 105, 115, 32, 97, 110, 32, 76, 76, 77]
```

**Why both work?** ASCII characters map directly to their Unicode code points (0-127), which UTF-8 represents with single bytes.

### Unicode and Multi-Byte Characters

```python
text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄"
tokens = text.encode("utf-8")
tokens = list(map(int, tokens))

# Text length: 111 characters
# Bytes length: 616 bytes
```

**Why 616 bytes for 111 characters?**
- Full-width characters: 3 bytes each
- Emoji: 4 bytes each
- Regional flags: 8+ bytes each (combination of emoji + combining characters)

**Key insight**: UTF-8 uses variable-length encoding:
- ASCII (0-127): 1 byte
- Latin extended: 2 bytes
- Common Unicode: 3 bytes
- Emoji, rare characters: 4 bytes

---

## PART 4: BPE ALGORITHM DEEP DIVE

### Step 1: Counting Consecutive Pairs

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts
```

**How it works**:
```python
tokens = [104, 101, 108, 108, 111, 44, 32]  # "hello, "
stats = get_stats(tokens)
# stats = {
#     (104, 101): 1,  # h-e
#     (101, 108): 1,  # e-l
#     (108, 108): 1,  # l-l (consecutive!)
#     (108, 111): 1,  # l-o
#     (111, 44): 1,   # o-,
#     (44, 32): 1     # ,-space
# }
```

**Finding the most frequent pair**:
```python
top_pair = max(stats, key=stats.get)
# top_pair = (108, 108)  # "ll" appears once
```

### Step 2: Merging Pairs

```python
def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids
```

**Example merge**:
```python
ids = [5, 6, 6, 7, 9, 1]
result = merge(ids, (6, 7), 99)
# result = [5, 6, 99, 9, 1]
#           ^  ^   ^^        Replaced (6,7) with 99
```

**Another example**:
```python
tokens = [104, 101, 108, 108, 111, 32, 116, 104, 105, 115]  # "hello this"
tokens2 = merge(tokens, (101, 32), 256)
# tokens2 = [104, 256, 108, 108, 111, 116, 104, 105, 115]
#                  ^^^
#                  Replaced "e " (101, 32) with new token 256
```

### Step 3: Iterative Merging

```python
vocab_size = 276
num_merges = vocab_size - 256  # 20 merges
ids = list(tokens)  # Copy of original bytes

merges = {}  # (int, int) -> int
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f"merging {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx
```

**Example output**:
```
merging (101, 32) into a new token 256
merging (240, 159) into a new token 257
merging (226, 128) into a new token 258
merging (105, 110) into a new token 259
...
merging (32, 262) into a new token 275
```

**Compression achieved**:
```python
print(f"tokens length: {len(tokens)}")   # 616 bytes
print(f"ids length: {len(ids)}")         # 451 tokens
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")
# Output: compression ratio: 1.37X
```

**What each merge means**:
- Token 256: bytes [101, 32] = "e "
- Token 257: bytes [240, 159] = emoji prefix
- Token 258: bytes [226, 128] = another emoji prefix
- Token 259: bytes [105, 110] = "in"
- ...

**Result**: 276-token vocabulary that compresses text by 37%

---

## PART 5: TRAINING A TOKENIZER

### Starting Vocabulary

```python
vocab = {idx: bytes([idx]) for idx in range(256)}
# vocab = {
#     0: b'\x00',
#     1: b'\x01',
#     ...
#     97: b'a',
#     98: b'b',
#     ...
#     255: b'\xff'
# }
```

**Initial vocab size**: 256 tokens (all possible byte values)

### Training Process

**Input**: Text corpus
```python
text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ ... (long text)"
tokens = list(text.encode("utf-8"))  # 616 bytes
```

**Training loop** (simplified):
```python
def train(text, vocab_size=276, verbose=False):
    num_merges = vocab_size - 256
    ids = list(text.encode("utf-8"))
    merges = {}

    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        idx = 256 + i
        if verbose:
            print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx

    return merges, ids

merges, compressed = train(text, vocab_size=276, verbose=True)
```

### Building Vocabulary from Merges

```python
def build_vocab(merges):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    return vocab

vocab = build_vocab(merges)
# vocab now has 276 entries:
# - 0-255: individual bytes
# - 256-275: merged byte sequences
```

**Example vocab entry**:
```python
vocab[256] = bytes([101, 32])  # "e "
vocab[259] = bytes([105, 110])  # "in"
vocab[261] = bytes([97, 110])   # "an"
```

### Compression Ratio Calculation

```python
original_bytes = len(text.encode("utf-8"))  # 616
compressed_tokens = len(compressed)         # 451
compression_ratio = original_bytes / compressed_tokens

print(f"Original: {original_bytes} bytes")
print(f"Compressed: {compressed_tokens} tokens")
print(f"Compression ratio: {compression_ratio:.2f}X")
```

**Typical compression ratios**:
- Small vocab (276 tokens): 1.3X - 1.4X
- Medium vocab (10,000 tokens): 1.5X - 1.7X
- Large vocab (50,000 tokens): 1.8X - 2.0X

---

## PART 6: ENCODING & DECODING

### Encoding (Text → Tokens)

**Goal**: Convert text to list of token IDs using trained merges

```python
def encode(text, merges):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens
```

**Step-by-step example**:
```python
text = "hello"
tokens = list(text.encode("utf-8"))
# tokens = [104, 101, 108, 108, 111]

# Iteration 1:
# stats = {(104,101):1, (101,108):1, (108,108):1, (108,111):1}
# merges contains (108,108): 256 (假设)
# min pair = (108,108) with priority 256
# tokens = merge([104,101,108,108,111], (108,108), 256)
# tokens = [104, 101, 256, 111]

# Iteration 2:
# stats = {(104,101):1, (101,256):1, (256,111):1}
# No more merges available in merges dict
# break

return [104, 101, 256, 111]
```

**Greedy approach**: Always merge highest-priority pair first
- Priority determined by merge order (earlier merges = lower token ID = higher priority)
- Ensures deterministic encoding

### Decoding (Tokens → Text)

**Goal**: Convert token IDs back to original text

```python
def decode(ids, vocab):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text
```

**Step-by-step example**:
```python
ids = [104, 101, 256, 111]
vocab = build_vocab(merges)  # From training
# vocab[256] = bytes([108, 108])  # "ll"

tokens = b"".join([vocab[104], vocab[101], vocab[256], vocab[111]])
# tokens = b'h' + b'e' + b'll' + b'o'
# tokens = b'hello'

text = tokens.decode("utf-8", errors="replace")
# text = "hello"

return "hello"
```

**Error handling**: `errors="replace"`
- If invalid UTF-8 byte sequence encountered, replaces with replacement character
- Prevents crashes on corrupted data

### Bidirectional Verification

```python
text = "Hello world! 안녕하세요 🎉"

encoded = encode(text, merges)
decoded = decode(encoded, vocab)

print(f"Original: {text}")
print(f"Match: {text == decoded}")
# Output: Match: True
```

**Lossless property**: `decode(encode(text)) == text` always holds (for valid UTF-8)

---

## PART 7: TIKTOKEN VS SENTENCEPIECE

### Tiktoken (OpenAI's Approach)

**Process**:
```
Text "Hello 안녕하세요"
  ↓
UTF-8 Encode → [72, 101, 108, 108, 111, 32, 236, 149, 136, ...]
  ↓
BPE on bytes → merges common byte sequences
  ↓
Tokens → [15496, 11, 9171, 299, 617, ...]
```

**Characteristics**:
- Encodes to UTF-8 bytes first
- Runs BPE on byte sequences (0-255 range)
- All tokens represent byte sequences
- Clean, predictable behavior
- Used by OpenAI (GPT-2, GPT-3, GPT-4)

**Advantages**:
- No special handling for rare characters
- Direct byte-level control
- Easier to debug
- Guaranteed lossless encoding

**Example**: GPT-2 vs GPT-4
```python
import tiktoken

# GPT-2 (does not merge spaces)
enc = tiktoken.get_encoding("gpt2")
print(enc.encode("    hello world!!!"))
# Output: [220, 220, 220, 220, 23748, 995, 10185]
#         ^^^^ ^^^^ ^^^^ ^^^^ spaces as individual tokens

# GPT-4 (merges spaces)
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode("    hello world!!!"))
# Output: [262, 24748, 1917, 12340]
#         ^^^ merged spaces token
```

### Sentencepiece (Google's Approach)

**Process**:
```
Text "Hello 안녕하세요"
  ↓
Unicode Code Points → [U+0048, U+0065, U+006C, ... U+C548, U+B155, ...]
  ↓
BPE on code points → merges common sequences
  ↓
Rare characters → UNK or byte fallback
  ↓
Tokens → [15496, 11, 9171, 299, 617, ...]
```

**Characteristics**:
- Runs BPE directly on Unicode code points (not bytes)
- Uses `character_coverage` parameter (e.g., 0.99995)
- Rare codepoints get UNK token OR byte fallback
- More complex configuration
- Used by Llama, Mistral, and many others

**Advantages**:
- Better compression for high-resource languages
- Flexible rare character handling
- Production-ready with many features

### Comparison Summary

| Aspect | Tiktoken | Sentencepiece |
|--------|-----------|---------------|
| Primary unit | UTF-8 bytes | Unicode code points |
| Rare characters | Handled naturally | UNK or byte fallback |
| Configuration | Simple | Hundreds of options |
| Used by | OpenAI | Meta, Mistral, Google |
| Philosophy | Clean, predictable | Flexible, production-ready |
| Byte fallback | Always uses bytes | Optional (`byte_fallback`) |

---

## PART 8: SENTENCEPIECE DEEP DIVE

### Why Sentencepiece is Popular

**Key reasons**:
- Efficient for BOTH training and inference BPE tokenizers
- Used in both Llama and Mistral series (major open-source models)
- Production-ready implementation with extensive features
- Handles edge cases (rare characters, normalization, splitting)

### Key Differences from Tiktoken

**1. Primary unit: Code points, not bytes**
```
Tiktoken: "안녕하세요" → UTF-8 bytes [236,149,136,235,133,149, ...]
Sentencepiece: "안녕하세요" → Code points [U+C548, U+B155, U+D559, ...]
```

**2. Rare character handling**
```python
character_coverage=0.99995  # 99.995% of characters covered
```
- Characters appearing less frequently than threshold are rare
- Rare chars go to UNK token OR byte fallback

**3. Byte fallback mechanism**
```python
byte_fallback=True
```
- Rare characters encoded as UTF-8 bytes
- Each byte becomes a token
- Example: Rare emoji → `<0xF0>` `<0x9F>` `<0x8C>` `<0x81>`

**4. Extensive configuration** (hundreds of hyperparameters)

### Llama 2 Configuration Example

```python
import sentencepiece as spm
import os

options = dict(
    # Input spec
    input="toy.txt",
    input_format="text",

    # Output spec
    model_prefix="tok400",

    # Algorithm spec
    model_type="bpe",
    vocab_size=400,

    # Normalization
    normalization_rule_name="identity",  # Turn off normalization
    remove_extra_whitespaces=False,
    input_sentence_size=200000000,
    max_sentence_length=4192,
    seed_sentencepiece_size=1000000,
    shuffle_input_sentence=True,

    # Rare word treatment
    character_coverage=0.99995,
    byte_fallback=True,

    # Merge rules
    split_digits=True,
    split_by_unicode_script=True,
    split_by_whitespace=True,
    split_by_number=True,
    max_sentencepiece_length=16,
    add_dummy_prefix=True,
    allow_whitespace_only_pieces=True,

    # Special tokens
    unk_id=0,  # UNK token MUST exist
    bos_id=1,  # Beginning of sequence
    eos_id=2,  # End of sequence
    pad_id=-1,  # Padding (disabled)

    # Systems
    num_threads=os.cpu_count(),
)

# Train
spm.SentencePieceTrainer.Train(**options)

# Load
sp = spm.SentencePieceProcessor()
sp.load('tok400.model')
```

### SentencePiece Encoding Example

```python
ids = sp.encode("hello 안녕하세요")
print(ids)
# Output: [362, 378, 361, 372, 358, 362, 239, 152, 139,
#          238, 136, 152, 240, 152, 155, 239, 135, 187,
#          239, 157, 151]

pieces = [sp.id_to_piece(idx) for idx in ids]
print(pieces)
# Output: ['▁', 'h', 'e', 'l', 'lo', '▁', '<0xEC>', '<0x95>',
#          '<0x88>', '<0xEB>', '<0x85>', '<0x95>', '<0xED>',
#          '<0x95>', '<0x98>', '<0xEC>', '<0x84>', '<0xB8>',
#          '<0xEC>', '<0x9A>', '<0x94>']
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#          Korean characters as hex bytes (due to rarity or byte_fallback)
```

**Understanding the output**:
- `▁` : Sentencepiece's space token
- `h`, `e`, `l`, `lo`: Character-level tokens for English
- `<0xEC>`, `<0x95>`, ...: Hex bytes representing Korean characters
- The Korean characters are treated as rare and fall back to byte encoding

### Character Coverage Explained

**Definition**: Percentage of unique characters in training data covered by vocabulary

```python
character_coverage=0.99995  # 99.995% coverage
```

**How it works**:
- Count all unique characters in training data
- Sort by frequency
- Include most frequent characters until coverage threshold reached
- Remaining 0.005% of characters are "rare"
- Rare characters: UNK token OR byte fallback

**Impact**:
```
character_coverage=0.99    → Fewer tokens, more UNK/bfallback
character_coverage=0.99995 → More tokens, fewer UNK/bfallback
character_coverage=1.0      → All characters, vocabulary explosion
```

**Trade-off**:
- High coverage: Larger vocabulary, better accuracy, more memory
- Low coverage: Smaller vocabulary, faster, more UNK tokens

### Key Sentencepiece Features

**1. Pre-tokenization rules**
```python
split_by_unicode_script=True   # Split different scripts (Latin vs Korean vs Arabic)
split_by_whitespace=True      # Split on whitespace
split_by_number=True         # Separate numbers from text
split_digits=True            # Split digits within numbers
```

**2. Maximum piece length**
```python
max_sentencepiece_length=16
```
- Limits BPE merge length
- Prevents extremely long tokens
- Balances compression vs. efficiency

**3. Dummy prefix**
```python
add_dummy_prefix=True
```
- Adds invisible start token to every sentence
- Helps model learn sentence boundaries
- Equivalent to GPT's learned positional embeddings

**4. Special tokens**
```python
unk_id=0   # Unknown token (for rare/out-of-vocab)
bos_id=1   # Beginning of sequence
eos_id=2   # End of sequence
pad_id=-1   # Padding (disabled with -1)
```

---

## PART 9: SPECIAL TOKENS

### Why Special Tokens Exist

**Purpose**: Provide control signals to the model

**Common special tokens**:
- `<|endoftext|>` or `<eos>`: End of text/sequence
- `<|startoftext|>` or `<bos>`: Beginning of sequence
- `<pad>`: Padding for batch processing
- `<unk>`: Unknown character (rare/out-of-vocab)
- `<mask>`: Masked language modeling (BERT-style)
- `<sep>`: Separator between segments

### GPT-2 Special Tokens

```python
import json

with open('encoder.json', 'r') as f:
    encoder = json.load(f)

len(encoder)  # 50257
```

**Breakdown**:
- 256: Raw byte tokens (0-255)
- 50,000: Merged tokens from BPE training
- 1: Special token `<|endoftext|>`

**The only special token**:
```python
encoder['<|endoftext|>']  # 50256
```

**Usage**:
- Marks end of text during training
- Prevents model from generating indefinitely
- Sometimes used to separate training examples

### Handling Special Tokens in Training

**Never merge special tokens**:
```python
# During BPE training, exclude special token sequences from merges
special_tokens = {'<|endoftext|>', '<|startoftext|>'}
# These byte sequences never get merged
```

**Why?**
- Special tokens must remain intact
- Model needs to recognize them as distinct signals
- If merged, model loses control ability

### Special Token IDs in Production

**Typical ID assignments**:
```python
special_tokens = {
    '<pad>': 0,        # Padding
    '<unk>': 1,        # Unknown
    '<bos>': 2,        # Beginning of sequence
    '<eos>': 3,        # End of sequence
    '<mask>': 4,       # Mask
    # ... other special tokens
    # Regular tokens start from higher IDs
}
```

**Impact on vocab size**:
```
vocab_size = regular_tokens + special_tokens
```

---

## PART 10: PRACTICAL CONSIDERATIONS

### Vocabulary Size Trade-offs

**Small vocabulary (e.g., 1,000 tokens)**:
- ✅ Less memory (1,000 × 384 = 384K parameters)
- ✅ Faster training
- ✅ Simpler model
- ❌ Longer sequences (more computation per token)
- ❌ Poor compression ratio (~1.1X - 1.2X)
- ❌ May need more tokens for rare words

**Large vocabulary (e.g., 50,000 tokens)**:
- ✅ Better compression (~1.5X - 2.0X)
- ✅ Shorter sequences
- ✅ Better handling of rare words
- ❌ More memory (50,000 × 384 = 19M parameters)
- ❌ Slower training
- ❌ More special cases

### GPT-2 vs GPT-4 Differences

**Space handling**:
```python
# GPT-2: spaces not merged
enc = tiktoken.get_encoding("gpt2")
enc.encode("    hello")
# [220, 220, 220, 220, 23748]
#  ^^^^ ^^^^ ^^^^ ^^^^ 4 space tokens

# GPT-4: spaces merged
enc = tiktoken.get_encoding("cl100k_base")
enc.encode("    hello")
# [262, 24748]
#  ^^^ 1 merged space token
```

**Impact**:
- GPT-4 more efficient for text with repeated spaces (e.g., code indentation)
- GPT-4 better at maintaining formatting

### Pre-tokenization Patterns

**GPT-2 pre-tokenization regex**:
```python
import regex as re

gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
```

**Breakdown**:
```
's|'t|'re|'ve|'m|'ll|'d    → Contractions split separately
 ?\p{L}+                     → Optional space + letters (Unicode-aware)
 ?\p{N}+                     → Optional space + numbers (Unicode-aware)
 ?[^\s\p{L}\p{N}]+          → Optional space + other characters
 \s+(?!\S)                    → Whitespace not followed by non-whitespace
 \s+                           → Any whitespace
```

**Example**:
```python
re.findall(gpt2pat, "Hello've world123 how's are you!!!?")
# Output: ['Hello', "'ve", ' world', '123', ' how', "'s", ' are', ' you', '!!!?']
```

**For Python code**:
```python
example = """
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
"""
re.findall(gpt2pat, example)
# ['\n', 'for', ' i', ' in', ' range', '(', '1', ',', ' 101', '):', '\n   ',
#  ' if', ' i', ' %', ' 3', ' ==', ' 0', ' and', ' i', ' %', ' 5', ' ==', ' 0', ':', '\n       ',
#  ' print', '("', 'FizzBuzz', '")', '\n']
```

**Why pre-tokenization?**
- Improves tokenization quality
- Handles edge cases (contractions, numbers, symbols)
- Makes BPE more efficient

### Compression Ratios

**Typical ranges**:
```
ASCII-only text:    1.2X - 1.5X
English text:       1.3X - 1.7X
Multilingual text:   1.4X - 1.8X
Code:              1.1X - 1.4X (often worse due to symbols/indentation)
```

**Example from notebook**:
```python
text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ ... (533 chars)"
tokens = list(text.encode("utf-8"))  # 616 bytes
compressed = train_and_encode(text, vocab_size=276)

print(f"Original: {len(tokens)} bytes")
print(f"Compressed: {len(compressed)} tokens")
print(f"Compression: {len(tokens) / len(compressed):.2f}X")
# Original: 616 bytes
# Compressed: 451 tokens
# Compression: 1.37X
```

### Tokenization Weirdness Examples

**1. Arithmetic**:
```python
# Numbers split unpredictably
"127 + 677"      → tokens for "127", "+", "677"
"1275 + 6773"    → tokens for "1275", "+", "6773"
# Different tokens for "127" vs "1275"!
```

**2. Case sensitivity**:
```python
"Egg."   → tokens for "E", "gg", "."
"egg."   → tokens for "e", "gg", "."
"EGG."   → tokens for "E", "G", "G", "."
# All different due to case!
```

**3. Trailing whitespace**:
```python
"hello"  → tokens for "hello"
"hello " → tokens for "hello", " "
# Trailing space adds token!
```

**4. Rare strings**:
```python
"SolidGoldMagikarp"  → May not exist in vocab, split character by character
"Hello world"       → Common, likely efficient tokens
```

---

## PART 11: COMPLETE DATA FLOW

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                           │
│  Input: Text Corpus "Hello world..."                       │
│    ↓                                                      │
│  UTF-8 Encode: [72, 101, 108, 108, 111, 32, ...]      │
│  Shape: (N,) where N = number of bytes                    │
│    ↓                                                      │
│  Count Pairs: {(104,101):5, (101,108):3, ...}            │
│    ↓                                                      │
│  Find Top Pair: max(stats, key=stats.get)                  │
│    ↓                                                      │
│  Merge: merge(ids, top_pair, new_idx)                      │
│  Shape: (N-k,) where k = num_merges                     │
│    ↓                                                      │
│  Record: merges[top_pair] = new_idx                       │
│    ↓                                                      │
│  Repeat: Until vocab_size reached                           │
│    ↓                                                      │
│  Build Vocab: {0:b'\x00', ..., 255:b'\xff',             │
│                256:b'e ', 257:b'in', ...}                 │
│  Shape: {idx: bytes} dict                                │
│    ↓                                                      │
│  Output: merges, vocab saved to file                       │
│                                                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    ENCODING PHASE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                           │
│  Input: Text "Hello world"                              │
│    ↓                                                      │
│  UTF-8 Encode: [72, 101, 108, 108, 111, 32, 119, ...]   │
│  Shape: (13,)                                           │
│    ↓                                                      │
│  Greedy Merge:                                            │
│    ├─ Get stats: {(104,101):1, (101,108):1, ...}         │
│    ├─ Find mergeable pair: min(stats, key=priority)        │
│    ├─ Merge: merge(tokens, pair, idx)                     │
│    └─ Repeat: Until no more merges                        │
│  Shape: (8,)  → compressed from 13 to 8 tokens          │
│    ↓                                                      │
│  Output: [15496, 11, 9171, 299, 617, ...]            │
│  Shape: (M,) where M = compressed token count             │
│                                                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    DECODING PHASE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                           │
│  Input: Token IDs [15496, 11, 9171, ...]               │
│  Shape: (M,)                                            │
│    ↓                                                      │
│  Lookup Bytes: [b'Hello', b' ', b'world', ...]             │
│  Shape: (bytes...)                                       │
│    ↓                                                      │
│  Join: b'Hello world'                                    │
│    ↓                                                      │
│  UTF-8 Decode: "Hello world"                              │
│    ↓                                                      │
│  Output: Text (lossless)                                  │
│                                                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                 LLM INTEGRATION PHASE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                           │
│  Input: Text "The quick brown fox"                        │
│    ↓                                                      │
│  Tokenizer.encode() → [464, 2068, 7586, 21831, ...]     │
│  Shape: (T,) where T = sequence length                   │
│    ↓                                                      │
│  Token Embedding: (B, T, C) = (batch, seq, embed_dim)    │
│    ↓                                                      │
│  Position Embedding: (B, T, C)                           │
│    ↓                                                      │
│  Transformer Blocks: (B, T, C) unchanged                   │
│    ↓                                                      │
│  Output Projection: (B, T, vocab_size)                   │
│    ↓                                                      │
│  LLM Output: Logits for next token                        │
│                                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## PART 12: KEY INSIGHTS

1. **Tokenizer is independent**: Completely separate module from LLM, can be trained on different data
2. **LLM never sees text**: Only ever processes token IDs, never raw text
3. **Compression efficiency**: BPE reduces sequence length by 30-50% typically
4. **Trade-offs**: Larger vocab = shorter sequences but bigger embedding matrices
5. **Unicode handling**: tiktoken (bytes) vs sentencepiece (code points) have fundamentally different approaches
6. **Special tokens**: Critical for control signals (padding, start/end of sequence, unknown handling)
7. **Greedy encoding**: Always merge highest-priority pair first (deterministic)
8. **Lossless property**: `decode(encode(text)) == text` always holds for valid UTF-8
9. **Pre-tokenization matters**: Regex patterns before BPE improve tokenization quality
10. **Compression varies by language**: English ~1.5X, multilingual ~1.7X, code ~1.2X
11. **Weirdness source**: Most LLM quirks stem from tokenization, not the model itself
12. **Byte-level safety**: UTF-8 encoding guarantees any text can be tokenized
13. **Vocabulary growth**: From 256 bytes to 50K+ tokens through iterative merging
14. **Deterministic training**: Same corpus + same vocab_size = same merges

---

## PART 13: RECOMMENDATIONS

### Don't Brush Off Tokenization

**Security and safety issues**:
- Input validation depends on tokenization
- Adversarial attacks exploit tokenization weaknesses
- Prompt injection leverages special token handling

**Performance issues**:
- Poor tokenization = slower inference (more tokens)
- Bad compression = wasted compute
- Rare tokens = poor generalization

### For Production Applications

**Option 1: Re-use existing tokenizers**
```python
import tiktoken

# Use GPT-4 tokenizer (already trained on massive corpus)
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("your text here")
```

**Advantages**:
- Already optimized
- Tested at scale
- Handles edge cases
- No training needed

**Option 2: Train custom tokenizer**
```python
# Use sentencepiece for custom training
import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input="your_corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=30000,
    model_type="bpe",
    character_coverage=0.99995,
    byte_fallback=True
)
```

**Advantages**:
- Optimized for your data
- Domain-specific tokens
- Better compression for your use case

**Option 3: Use tiktoken for training**
```python
# Tiktoken approach (cleaner, more predictable)
from tokenizer import Tokenizer

tokenizer = Tokenizer(vocab_size=50000)
tokenizer.train("your_corpus.txt", verbose=True)
tokens = tokenizer.encode("your text here")
```

### When to Use Which Approach

| Use case | Recommended tokenizer |
|-----------|---------------------|
| General purpose | GPT-4 tiktoken (cl100k_base) |
| English-only | GPT-2 tiktoken (gpt2) |
| Multilingual | Custom sentencepiece |
| Code-heavy | Custom tokenizer with code patterns |
| Research/education | Implement BPE from scratch |

### Best Practices

1. **Always use UTF-8 encoding**: Ensures any text can be tokenized
2. **Include special tokens**: Essential for control signals
3. **Handle rare characters**: Use byte fallback or UNK token
4. **Test on your data**: Verify compression ratio and tokenization quality
5. **Monitor token count**: Affects cost and latency
6. **Consider pre-tokenization**: Improves quality for specific domains
7. **Document your tokenizer**: Save merges, vocab, configuration

### Future Directions

**The ultimate goal**: Eliminate tokenization entirely
- Direct character-level or byte-level modeling
- More elegant, no weirdness
- But: Currently impractical (too expensive computationally)

**Current state**: "Eternal glory to whoever can delete the need for tokenization"

---

## PART 14: VOCABULARY SIZE

### Q: What Should Be Vocab Size?

**Key factors to consider**:

1. **Compression efficiency**
   - Larger vocab = fewer tokens per sequence
   - Better compression ratio (1.2X → 2.0X)

2. **Memory footprint**
   - Embedding matrix: vocab_size × embedding_dim parameters
   - Directly affects GPU memory and training time

3. **Coverage**
   - Must represent all characters + common subsequences
   - Rare words: balance between specific tokens vs. byte sequences

4. **Target language**
   - English: 10K-30K tokens sufficient
   - Multilingual: 50K-100K tokens needed
   - Code-heavy: May need fewer tokens (limited vocabulary)

**Typical vocab sizes in production**:

| Model | Vocab Size | Notes |
|--------|-------------|--------|
| GPT-2 | 50,257 | 256 bytes + 50K merges + 1 special |
| GPT-3 | 50,257 | Same as GPT-2 |
| GPT-4 | 100,277 | Nearly 2x GPT-2 |
| Llama 2 | 32,000 | Smaller, multilingual |
| Claude | 100,000 | Similar to GPT-4 |
| Our demo | 276 | For educational purposes |

**Trade-offs visualization**:
```
Small vocab (e.g., 1,000):
  ✓ Memory: 1,000 × 384 = 384K parameters
  ✓ Training: Fast
  ✓ Inference: Fast
  ✗ Sequence length: Long (1.2X - 1.3X compression)
  ✗ Rare words: Many character-level tokens

Medium vocab (e.g., 30,000):
  ✓ Memory: 30,000 × 384 = 11.5M parameters
  ✓ Training: Moderate
  ✓ Inference: Moderate
  ✓ Sequence length: Short (1.5X - 1.7X compression)
  ✓ Rare words: Some subword tokens

Large vocab (e.g., 100,000):
  ✓ Memory: 100,000 × 384 = 38.4M parameters
  ✗ Training: Slow
  ✗ Inference: Slow
  ✓ Sequence length: Very short (1.8X - 2.0X compression)
  ✓ Rare words: Many specific tokens
```

**Guidelines for choosing vocab_size**:

| Situation | Recommended vocab_size | Rationale |
|-----------|----------------------|------------|
| Educational demo | 276 - 1,000 | Fast training, clear to understand |
| English text | 10,000 - 30,000 | Good compression, manageable memory |
| Multilingual | 30,000 - 50,000 | More tokens needed for diversity |
| Large corpus | 50,000 - 100,000 | Justified by data size |
| Limited compute | 1,000 - 10,000 | Faster training/inference |
| Production (general) | 30,000 - 50,000 | Balance of compression and efficiency |

### Q: How Can I Increase Vocab Size?

**In our tokenizer implementation**:

```python
from tokenizer import Tokenizer

# Change vocab_size parameter
tokenizer = Tokenizer(vocab_size=10000)  # Was 276, now 10,000
tokenizer.train(text, verbose=True)
```

**What happens internally**:
```python
num_merges = vocab_size - 256
# For vocab_size=276: num_merges = 20
# For vocab_size=10000: num_merges = 9744
```

**More merges mean**:
- More iterations during training
- Learns longer/more common subsequences
- Better compression ratio

**Example impact**:
```python
# Vocab size: 276 tokens
text = "Many common characters, including numerals..."
compressed = tokenizer.encode(text)
compression_ratio = len(text.encode("utf-8")) / len(compressed)
# Compression ratio: 1.37X

# Vocab size: 1,000 tokens
tokenizer = Tokenizer(vocab_size=1000)
tokenizer.train(text)
compressed = tokenizer.encode(text)
compression_ratio = len(text.encode("utf-8")) / len(compressed)
# Compression ratio: 1.50X

# Vocab size: 10,000 tokens
tokenizer = Tokenizer(vocab_size=10000)
tokenizer.train(text)
compressed = tokenizer.encode(text)
compression_ratio = len(text.encode("utf-8")) / len(compressed)
# Compression ratio: 1.80X
```

### In LLM Architecture (Connecting to mental_model.md)

**Updating embedding table**:
```python
# Original: vocab_size=65, n_embd=384
self.token_embedding_table = nn.Linear(vocab_size, n_embd)
# Shape: (65, 384) = 24,960 parameters

# Updated: vocab_size=10000, n_embd=384
self.token_embedding_table = nn.Linear(vocab_size, n_embd)
# Shape: (10000, 384) = 3,840,000 parameters (153x increase!)
```

**Updating output projection**:
```python
# Original
self.lm_head = nn.Linear(n_embd, vocab_size)
# Shape: (384, 65) = 24,960 parameters

# Updated
self.lm_head = nn.Linear(n_embd, vocab_size)
# Shape: (384, 10000) = 3,840,000 parameters (153x increase!)
```

**Total parameter increase**:
```
Original: 24,960 + 24,960 = 49,920 parameters
Updated: 3,840,000 + 3,840,000 = 7,680,000 parameters
Increase: 153x (from 50K to 7.68M parameters)
```

### Steps to Increase Vocab Size

**1. Train new tokenizer**:
```python
# Train with larger vocab_size on corpus
tokenizer = Tokenizer(vocab_size=10000)
tokenizer.train(corpus_text, verbose=True)

# Save merges and vocab
import pickle
with open('tokenizer_merges.pkl', 'wb') as f:
    pickle.dump(tokenizer.merges, f)
with open('tokenizer_vocab.pkl', 'wb') as f:
    pickle.dump(tokenizer.vocab, f)
```

**2. Update LLM architecture**:
```python
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size=10000, n_embd=384):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
```

**3. Retrain or fine-tune**:
```python
# Option A: Retrain from scratch (recommended)
model = GPTLanguageModel(vocab_size=10000)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
train(model, optimizer, train_data)  # Train from scratch

# Option B: Fine-tune (partial retraining)
# Not recommended - embeddings change shape, need careful handling
```

**4. Verify compression improvement**:
```python
test_text = "The quick brown fox jumps over the lazy dog."

old_compression = 1.37  # From vocab_size=276
new_compression = len(test_text.encode("utf-8")) / len(tokenizer.encode(test_text))

print(f"Old compression: {old_compression}X")
print(f"New compression: {new_compression}X")
print(f"Improvement: {new_compression / old_compression:.2f}X")
```

### When to Increase Vocab Size

**Increase when**:
1. **Compression ratio is poor**: < 1.2X indicates vocab too small
2. **Training data has many rare words**: Domain-specific terminology
3. **Target language has complex morphology**: Agglutinative languages (Finnish, Turkish)
4. **Memory/compute budget allows**: Can afford larger embeddings
5. **Sequence length is bottleneck**: Reducing token count speeds up training

**Keep small when**:
1. **Limited compute**: Smaller models train faster
2. **Memory constrained**: Embedded devices, inference on edge
3. **Simple domain**: Limited vocabulary (e.g., chemical formulas)
4. **Educational purposes**: Easier to understand with small vocab

### Practical Example: Scaling Up

```python
# Starting point (our implementation)
tokenizer_small = Tokenizer(vocab_size=276)
tokenizer_small.train(text, verbose=False)
print(f"Small vocab size: {len(tokenizer_small.vocab)}")  # 276
print(f"Compression: {1.37}X")

# Medium vocabulary
tokenizer_medium = Tokenizer(vocab_size=1000)
tokenizer_medium.train(text, verbose=False)
print(f"Medium vocab size: {len(tokenizer_medium.vocab)}")  # 1000
print(f"Compression: {1.50}X")

# Large vocabulary
tokenizer_large = Tokenizer(vocab_size=10000)
tokenizer_large.train(text, verbose=False)
print(f"Large vocab size: {len(tokenizer_large.vocab)}")  # 10000
print(f"Compression: {1.80}X")

# Production vocabulary (like GPT-2)
tokenizer_production = Tokenizer(vocab_size=50257)
tokenizer_production.train(large_corpus, verbose=True)
print(f"Production vocab size: {len(tokenizer_production.vocab)}")  # 50257
print(f"Compression: {1.95}X")
```

**Memory comparison** (assuming n_embd=384):
```
vocab_size=276:    276 × 384 × 2 = 212K parameters
vocab_size=1000:   1,000 × 384 × 2 = 768K parameters (3.6x increase)
vocab_size=10000:  10,000 × 384 × 2 = 7.68M parameters (36x increase)
vocab_size=50257:  50,257 × 384 × 2 = 38.6M parameters (182x increase!)
```

**Conclusion**: Vocab size is a critical hyperparameter with significant trade-offs. Choose based on your specific constraints and goals.
