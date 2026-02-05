# Complete GPT Architecture Deep Dive

I'll walk you through the entire GPT architecture step-by-step, covering everything: shapes, purpose, connections, and mathematics. Let's start from the beginning and build up understanding layer by layer.

---

## PART 1: THE BIG PICTURE (5-Second Mental Model)

```
Text "ROMEO:" → Numbers → Embeddings → 6 Transformer Blocks → Predictions → Next Character
```

**The core idea**: The model learns to predict the next character by looking at all previous characters and deciding which ones are important for the prediction.

---

## PART 2: COMPLETE DATA FLOW WITH SHAPES

Let's trace **ONE example** through the entire model. Say we have:
- Batch size: 2 (2 sequences being processed at once)
- Block size: 3 (shortened from 256 for clarity)
- We want to predict what comes after: `"ROMEO:"`

### Step 1: Input Preparation

**Raw input** (text):
```
Sequence 1: "ROMEO:"
Sequence 2: "JULIE:"
```

**Encode to numbers** (gpt.py:31-33):
```python
encode("ROMEO:") = [46, 47, 56, 42, 45, 1]  # Example indices
encode("JULIE:") = [40, 48, 42, 47, 45, 1]  # Example indices
```

**Create batch tensor** (gpt.py:46-51):
```python
idx = [
    [46, 47, 56],  # First 3 chars of "ROM"
    [40, 48, 42]   # First 3 chars of "JUL"
]
```

**Shape**: `(B=2, T=3)` - 2 sequences, each 3 characters long

---

### Step 2: Embeddings (gpt.py:178-180)

**Token Embeddings** - What each character **is**:
```python
tok_emb = self.token_embedding_table(idx)  # (B,T,C) = (2,3,384)
```

- `token_embedding_table`: Size `(65, 384)` - 65 characters in vocabulary, each gets a 384-dimensional vector
- Each index (like 46) gets looked up, returning a 384-dimensional vector
- Result: `(2,3,384)` - each position now has a 384-dimensional representation

**Position Embeddings** - Where each character **is**:
```python
pos_emb = self.position_embedding_table(torch.arange(T))  # (T,C) = (3,384)
```

- `position_embedding_table`: Size `(256, 384)` - 256 possible positions (block_size)
- `torch.arange(T) = [0, 1, 2]` - position indices
- Result: `(3,384)` - position 0, position 1, position 2 each have 384-dim vectors

**Combine them** (broadcasted addition):
```python
x = tok_emb + pos_emb  # (2,3,384)
```

**Broadcasting**: `(2,3,384) + (3,384)` → PyTorch automatically broadcasts `(3,384)` to `(2,3,384)` by copying it for each batch element

**Why addition works**: Both embeddings are in the same 384-dimensional space, so addition combines the information meaningfully.

**What `x` contains**: For each position in each sequence, we now have a 384-dim vector representing BOTH "what character is this" AND "where is it in the sequence"

---

### Step 3: Transformer Block 1 (gpt.py:133-148)

Each block has **2 parts**:
1. **Multi-Head Self-Attention** (communication)
2. **Feed-Forward Network** (computation)

Let's go through Block 1 in detail.

#### Block 1, Part A: Layer Norm 1

```python
x = x + self.sa(self.ln1(x))  # First, normalize x
```

**What LayerNorm does** (gpt.py:142):
```python
self.ln1 = nn.LayerNorm(n_embd)  # n_embd = 384
```

For each sample, independently:
```
normalized = (x - mean) / sqrt(variance + epsilon)
```

**Shape before**: `(2,3,384)`
**Shape after**: `(2,3,384)` - same shape, but normalized values

**Purpose**: Stabilizes training by keeping values in a reasonable range (roughly mean=0, std=1)

---

#### Block 1, Part B: Multi-Head Attention

```python
x = x + self.sa(self.l1n(x))  # Apply self-attention
```

Let's unpack `self.sa(x)`:

**MultiHeadAttention** creates **6 heads** in parallel (gpt.py:102-114):
```python
n_head = 6
head_size = n_embd // n_head = 384 // 6 = 64
```

**Each head independently** (gpt.py:71-99):
1. Creates Q, K, V from input `x`
2. Computes attention weights
3. Applies weights to values

**Inside ONE head**:

**Step B1: Create Q, K, V** (gpt.py:87-88):
```python
k = self.key(x)      # (B,T,hs) = (2,3,64)
q = self.query(x)    # (B,T,hs) = (2,3,64)
v = self.value(x)    # (B,T,hs) = (2,3,64)
```

- Each is a Linear layer: `nn.Linear(384, 64, bias=False)`
- Projects 384-dimensional input down to 64 dimensions
- Different heads have different projections (learn different patterns)

**Mathematically**: `q = x @ W_q` where `W_q` is a `(384, 64)` matrix

**What Q, K, V represent**:
- **Query (Q)**: "What am I looking for?" at each position
- **Key (K)**: "What do I contain?" at each position
- **Value (V)**: "What information do I share?" at each position

**Step B2: Compute Attention Scores** (gpt.py:90-92):
```python
wei = q @ k.transpose(-2, -1) * head_size**-0.5
```

**Matrix multiplication**:
```
q @ k.T
(2,3,64) @ (2,64,3) → (2,3,3)
```

- Each position in Q (3 positions) computes similarity with each position in K (3 positions)
- Result: `(2,3,3)` - for each sequence, a 3×3 matrix of "affinity scores"

**Concretely for one sequence**:
```
Position 0's query vs Position 0's key → score[0,0]
Position 0's query vs Position 1's key → score[0,1]
Position 0's query vs Position 2's key → score[0,2]
Position 1's query vs Position 0's key → score[1,0]
... and so on
```

**Why this matters**: High score = these two positions should "pay attention" to each other

**Scaling** (`* head_size**-0.5`):
- `head_size**-0.5 = 1/sqrt(64) = 1/8 = 0.125`
- Divides all scores by 8

**Why scale?**: Without scaling, large dot products cause softmax to become extremely peaked (almost all 0s except one 1). Scaling keeps gradients stable.

**Mathematics**: Variance of dot product of two unit vectors = dimension, so dividing by sqrt(dimension) keeps variance at 1.

**Step B3: Mask Future** (gpt.py:93):
```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
```

**Mask** (gpt.py:79):
```python
self.tril = torch.tril(torch.ones(block_size, block_size))
# Creates:
[[1, 0, 0, ...],
 [1, 1, 0, ...],
 [1, 1, 1, ...],
 ...]
```

**For T=3, the mask is**:
```
[[1, 1, 1],
 [1, 1, 1],
 [1, 1, 1]]
```

Wait, that's all 1s! But we use `self.tril[:T, :T] == 0`, so:
```
[[0, 0, 0],
 [0, 0, 0],
 [0, 0, 0]]
```

So **no masking happens** when T=3. Masking only happens when T < block_size.

**For T=2 (earlier in sequence)**:
```
self.tril[:2, :2] = [[1, 0], [1, 1]]
mask = [[False, True], [False, False]]
```

Position 0 can see position 0 (no mask)
Position 0 cannot see position 1 (mask = True → -inf)
Position 1 can see positions 0 and 1 (no mask)

**Purpose**: During training, we know the future (target characters), but the model shouldn't cheat by seeing them.

**Step B4: Softmax** (gpt.py:94):
```python
wei = F.softmax(wei, dim=-1)
```

Converts scores to probabilities that sum to 1.

**Example for one sequence, before masking**:
```
wei[0] = [5.0, 3.0, 2.0]  # Scores
softmax(wei[0]) = [0.84, 0.12, 0.04]  # Probabilities (sum to 1)
```

**After masking (position 0 can't see positions 1,2)**:
```
wei[0] = [5.0, -inf, -inf]
softmax(wei[0]) = [1.0, 0.0, 0.0]  # Only position 0!
```

**Why -inf → 0**: `exp(-inf) = 0`, so after softmax, those positions have 0 weight.

**Shape**: `(2,3,3)` - still attention weights, now as probabilities

**Step B5: Apply to Values** (gpt.py:97-98):
```python
v = self.value(x)  # (B,T,hs) = (2,3,64)
out = wei @ v      # (B,T,T) @ (B,T,hs) → (B,T,hs) = (2,3,64)
```

**Matrix multiplication**: Weighted average of values

**For one sequence, one position**:
```
out[0, 0] = wei[0, 0, 0] * v[0, 0] + wei[0, 0, 1] * v[0, 1] + wei[0, 0, 2] * v[0, 2]
```

If `wei[0, 0] = [0.5, 0.3, 0.2]`, then:
- Position 0 gets 50% of its own value
- Plus 30% of position 1's value
- Plus 20% of position 2's value

**Shape after one head**: `(2,3,64)` - same as input to the head

**Step B6: Concatenate 6 Heads** (gpt.py:112):
```python
out = torch.cat([h(x) for h in self.heads], dim=-1)
```

- Each head outputs `(2,3,64)`
- Concatenate 6 heads: `(2,3,64*6)` = `(2,3,384)`
- Concatenated dimension contains information from all 6 heads

**Step B7: Projection** (gpt.py:113):
```python
out = self.dropout(self.proj(out))  # (2,3,384) → (2,3,384)
```

`self.proj` is `nn.Linear(384, 384)` - learns to mix information from different heads.

**Purpose**: Heads might learn redundant or complementary patterns; projection learns the optimal combination.

**Shape after MultiHeadAttention**: `(2,3,384)` - same as input to the layer!

---

#### Block 1, Part C: Residual Connection

```python
x = x + self.sa(self.l1n(x))
```

- `x` was `(2,3,384)` before attention
- `self.sa(...)` outputs `(2,3,384)`
- Addition: `(2,3,384) + (2,3,384) → (2,3,384)`

**Purpose**:
- Gradients can flow unchanged through addition
- Model can learn to skip the attention if not needed (identity function)
- Enables training very deep networks

---

#### Block 1, Part D: Feed-Forward Network

```python
x = x + self.ffwd(self.ln2(x))
```

**First, normalize**:
```python
ln2_out = self.ln2(x)  # (2,3,384)
```

**Feed-forward** (gpt.py:117-130):
```python
self.ffwd = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),  # 384 → 1536
    nn.ReLU(),
    nn.Linear(4 * n_embd, n_embd),  # 1536 → 384
    nn.Dropout(dropout),
)
```

**Step D1: Expand**:
```python
layer1 = nn.Linear(384, 1536)
out1 = layer1(ln2_out)  # (2,3,1536)
```

**Why 4x expansion?**: Gives more capacity for non-linear transformations. Common design choice from original Transformer paper.

**Step D2: Non-linearity**:
```python
out2 = F.relu(out1)  # (2,3,1536)
```

- `relu(x) = max(0, x)` - sets negative values to 0
- Enables learning complex, non-linear patterns

**Step D3: Contract back**:
```python
layer2 = nn.Linear(1536, 384)
out3 = layer2(out2)  # (2,3,384)
```

Projects back to original dimension.

**Step D4: Dropout**:
```python
out4 = F.dropout(out3, p=0.2)  # (2,3,384)
```

Randomly sets 20% of values to 0 during training (prevents overfitting).

**Shape after Feed-Forward**: `(2,3,384)` - same as input!

---

#### Block 1, Part E: Second Residual

```python
x = x + self.ffwd(self.ln2(x))
```

Add feed-forward output to the original input.

**After Block 1**: `x` is still `(2,3,384)` but now contains information gathered from attention and processed by feed-forward.

---

### Step 4: Blocks 2-6 (gpt.py:157-159)

```python
self.blocks = nn.Sequential(
    *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]  # n_layer = 6
)
```

**What happens**: We repeat the same structure 6 times.

**Why multiple blocks?**
- Each block learns increasingly complex patterns
- Layer 1: Simple patterns (character co-occurrence)
- Layer 2: Combines layer 1 patterns
- Layer 3: Combines layer 2 patterns
- ...
- Layer 6: Hierarchical, abstract representations

**Analogy**:
- Layer 1 learns "a" often follows "th"
- Layer 2 learns "the" often precedes nouns
- Layer 3 learns phrases like "In the"
- Layer 6 learns complete grammatical structures

**Shape after all 6 blocks**: `(2,3,384)` - still the same!

---

### Step 5: Final Layer Norm (gpt.py:182)

```python
x = self.ln_f(x)  # (2,3,384)
```

Normalize one last time before output.

---

### Step 6: Output Projection (gpt.py:183)

```python
logits = self.lm_head(x)  # (B,T,vocab_size) = (2,3,65)
```

`self.lm_head` is `nn.Linear(384, 65)` - projects from embedding space to vocabulary space.

**For each position in each sequence**:
- Input: 384-dimensional vector
- Output: 65-dimensional vector (one score for each character in vocabulary)

**What logits mean**: Higher score = more likely next character

---

### Step 7: Loss Calculation (gpt.py:188-191)

```python
if targets is not None:
    B, T, C = logits.shape  # 2, 3, 65
    logits = logits.view(B * T, C)  # (2*3, 65) = (6, 65)
    targets = targets.view(B * T)  # (2*3,) = (6,)
    loss = F.cross_entropy(logits, targets)
```

**Flatten**: Combine batch and time dimensions
- `logits`: `(2,3,65)` → `(6,65)` - 6 predictions total
- `targets`: `(2,3)` → `(6,)` - 6 target characters

**Cross-entropy**:
- For each of the 6 positions, compute "how wrong" is the prediction
- Higher loss = worse prediction
- Lower loss = better prediction

**Shape of loss**: Scalar (single number)

---

### Step 8: Generation (gpt.py:195-210)

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]  # Keep only last block_size characters
        logits, loss = self(idx_cond)    # Forward pass
        logits = logits[:, -1, :]        # Take last position's logits
        probs = F.softmax(logits, dim=-1)  # Convert to probabilities
        idx_next = torch.multinomial(probs, num_samples=1)  # Sample
        idx = torch.cat((idx, idx_next), dim=1)  # Append
    return idx
```

**Step-by-step**:

1. **Crop to block_size**: If sequence is longer than 256, keep only last 256 characters
   ```
   idx = [[46, 47, 56, ..., 1]]  # 300 characters
   idx_cond = [[..., 1]]  # Last 256
   ```

2. **Forward pass**: Get predictions for all positions
   ```
   logits shape: (B=2, T=256, vocab=65)
   ```

3. **Take last position**: We only need predictions for the NEXT character (after the last one)
   ```
   logits = logits[:, -1, :]  # (2, 65) - one prediction per sequence
   ```

4. **Softmax**: Convert scores to probabilities
   ```
   probs shape: (2, 65) - sums to 1 for each sequence
   ```

5. **Sample**: Pick next character based on probabilities
   ```
   idx_next shape: (2, 1) - one character per sequence
   ```

6. **Append**: Add to sequence
   ```
   idx shape: (2, 257) - grew by 1 character
   ```

7. **Repeat**: Generate `max_new_tokens` characters

---

## PART 3: WHY EACH COMPONENT EXISTS

### Why Embeddings?

**Problem**: Neural networks work with numbers, not text. "R" as a number doesn't mean "letter R".

**Solution**: Embeddings learn semantic representations. Similar characters end up with similar vectors.

**Example**: After training, vowels might cluster together, consonants might cluster together.

### Why Positional Embeddings?

**Problem**: "the cat" and "cat the" use the same characters but mean different things. The model needs to know position.

**Solution**: Add position information to token embeddings.

**Why addition, not concatenation?**: Addition is more efficient and allows the model to learn interactions between identity and position.

### Why Attention?

**Problem**: In a long sequence, not all previous characters are equally important for predicting the next one.

**Solution**: Attention learns to "focus" on relevant characters.

**Example**: In "The quick brown fox jumps over the lazy dog", to predict "lazy", attention might focus on "the" and "dog", not "quick" or "fox".

### Why Multi-Head Attention?

**Problem**: Single attention pattern might not capture all types of relationships.

**Solution**: Multiple heads learn different patterns in parallel.

**Example**:
- Head 1: Focuses on grammatical structure (noun-verb agreement)
- Head 2: Focuses on semantic relationships (related words)
- Head 3: Focuses on long-range dependencies (pronoun references)

### Why Feed-Forward Networks?

**Problem**: Attention is linear (weighted sum). Complex patterns need non-linearity.

**Solution**: Feed-forward networks with ReLU introduce non-linearity.

**Analogy**: Attention gathers information, feed-forward processes it.

### Why Layer Normalization?

**Problem**: As data flows through layers, values can grow very large or very small, causing training instability.

**Solution**: Normalize each sample to have mean=0, std=1 at each layer.

**Why not BatchNorm?**: BatchNorm normalizes across the batch, which can be problematic for variable-length sequences.

### Why Residual Connections?

**Problem**: Deep networks suffer from vanishing gradients - hard to train.

**Solution**: Add input to output (`x = x + f(x)`).

**Why it works**: Gradients can flow unchanged through the addition, enabling very deep networks.

**Analogy**: If you have 100 layers, residuals let gradients jump directly from layer 100 to layer 99, then to 98, etc., without shrinking.

### Why Multiple Blocks?

**Problem**: Complex patterns require multiple levels of abstraction.

**Solution**: Stack blocks to create a hierarchy.

**Analogy**:
- CNNs: Early layers learn edges, middle layers learn shapes, later layers learn objects
- Transformers: Early blocks learn character patterns, later blocks learn grammar and semantics

### Why Dropout?

**Problem**: Models can overfit - memorize training data instead of learning generalizable patterns.

**Solution**: Randomly drop neurons during training forces model to be robust.

**Analogy**: Like practicing a sport with one eye closed - you learn to rely on all senses, not just one.

---

## PART 4: MATHEMATICS DEEP DIVE

### Scaled Dot-Product Attention Formula

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

**Breaking it down**:

1. **QK^T**: Query × Key transpose
   - Computes similarity between every query and every key
   - Matrix: (B, T, d_k) @ (B, d_k, T) → (B, T, T)
   - Each entry [i, j] is the similarity between query at position i and key at position j

2. **/ sqrt(d_k)**: Scale by square root of head size
   - Prevents softmax saturation
   - Keeps gradients stable

3. **softmax**: Convert to probabilities
   - exp(x) / sum(exp(x))
   - Ensures weights are non-negative and sum to 1

4. **V**: Multiply by values
   - Weighted sum of values
   - Each position's output is a combination of all values, weighted by attention

### Why Scaling Works

**Variance of dot product**:
- If Q and K are independent with unit variance
- Each element of QK^T has variance = d_k
- So QK^T grows with dimension, causing softmax to peak
- Dividing by sqrt(d_k) normalizes variance back to 1

**Example**:
```
Without scaling:
q @ k.T = [100, 90, 80, ...]
softmax([100, 90, 80, ...]) = [1.0, 0.0, 0.0, ...]  # All mass on first!

With scaling (divide by 8):
[100/8, 90/8, 80/8, ...] = [12.5, 11.25, 10, ...]
softmax([12.5, 11.25, 10, ...]) = [0.64, 0.25, 0.11, ...]  # Distributed!
```

### Feed-Forward Expansion Why 4x?

**Original Transformer paper** uses 4x, and it's become standard.

**Intuition**:
- Feed-forward happens after attention has gathered information
- You need more capacity to process rich, gathered information
- 4x is a good balance between compute and model capacity

**Math**:
```
Input: (B, T, 384)
After expansion: (B, T, 1536)
After ReLU: (B, T, 1536)
After contraction: (B, T, 384)
```

### Layer Normalization Formula

```
LN(x) = γ * ((x - μ) / sqrt(σ^2 + ε)) + β
```

**Where**:
- μ = mean of x (computed per sample, not per batch)
- σ^2 = variance of x
- γ = learnable scale parameter
- β = learnable shift parameter
- ε = small constant (prevents division by zero)

**Key point**: Mean and variance are computed across the feature dimension (C=384), not across the batch.

---

## PART 5: COMPLETE VISUAL FLOW

```
INPUT: "ROMEO:"
  ↓
ENCODE: [46, 47, 56, 42, 45, 1]
  ↓
BATCH: [[46, 47, 56], [40, 48, 42]]  (2, 3)
  ↓
TOKEN EMBEDDING: (2, 3, 384)
POSITION EMBEDDING: (3, 384) → (2, 3, 384) [broadcast]
  ↓
COMBINE: (2, 3, 384) = tok_emb + pos_emb
  ↓
BLOCK 1:
  ├─ LayerNorm: (2, 3, 384)
  ├─ MultiHeadAttention (6 heads):
  │   ├─ Head 1: (2, 3, 384) → (2, 3, 64) → (2, 3, 64)
  │   ├─ Head 2: (2, 3, 384) → (2, 3, 64) → (2, 3, 64)
  │   ├─ ... (6 heads total)
  │   └─ Concat & Project: (2, 3, 384)
  ├─ Residual: (2, 3, 384) + (2, 3, 384) = (2, 3, 384)
  ├─ LayerNorm: (2, 3, 384)
  ├─ FeedForward: (2, 3, 384) → (2, 3, 1536) → (2, 3, 384)
  └─ Residual: (2, 3, 384) + (2, 3, 384) = (2, 3, 384)
  ↓
BLOCKS 2-6: (repeat same structure)
  ↓
FINAL LAYERNORM: (2, 3, 384)
  ↓
OUTPUT PROJECTION: (2, 3, 65) - logits for each position
  ↓
SOFTMAX: (2, 3, 65) - probabilities
  ↓
SAMPLE: Pick next character
  ↓
OUTPUT: "ROMEO:" + next character
```

---

## PART 6: WHAT EACH LAYER LEARNS (Intuitive)

### Token Embeddings
- **Input**: Character indices
- **Output**: 384-dimensional vectors
- **Learns**: Semantic meaning of characters (vowels similar, consonants similar, etc.)

### Positional Embeddings
- **Input**: Position indices
- **Output**: 384-dimensional vectors
- **Learns**: "Position 5" means something different from "position 50"

### Attention Head 1
- **Input**: 384-dimensional vectors
- **Output**: 64-dimensional vectors
- **Learns**: Local patterns (e.g., "qu" often followed by vowel)

### Attention Head 2
- **Input**: 384-dimensional vectors
- **Output**: 64-dimensional vectors
- **Learns**: Medium-range patterns (e.g., verb agreement with subject)

### Attention Head 3
- **Input**: 384-dimensional vectors
- **Output**: 64-dimensional vectors
- **Learns**: Long-range dependencies (e.g., pronoun references)

### ... Heads 4-6
- Each learns different patterns

### Feed-Forward
- **Input**: 384-dimensional vectors (from attention)
- **Output**: 384-dimensional vectors (processed)
- **Learns**: How to combine and transform attention outputs

### Stacked Blocks
- **Layer 1**: Simple patterns
- **Layer 2**: Combines layer 1 patterns
- **Layer 3**: Combines layer 2 patterns
- **Layer 6**: Complex, abstract representations

---

## PART 7: CONCRETE EXAMPLE Walkthrough

Let's trace predicting the next character after "ROM:" in "ROMEO:"

**Initial state**:
```
Sequence: "ROM"
Encoded: [46, 47, 56]
Batch: [[46, 47, 56]]  (1, 3)
```

**After embeddings**:
```
tok_emb: [[v1, v2, v3]] where each vi is 384-dim vector
pos_emb: [[p0, p1, p2]] where each pi is 384-dim vector
x: [[v1+p0, v2+p1, v3+p2]]  (1, 3, 384)
```

**After Block 1**:
```
Position 0 (index 46, "R"):
  - Attention sees positions 0, 1, 2
  - Learns "R" often at start of words
  - Feed-forward processes this pattern

Position 1 (index 47, "O"):
  - Attention sees positions 0, 1
  - Learns "O" often follows "R"
  - Feed-forward processes

Position 2 (index 56, "M"):
  - Attention sees positions 0, 1, 2
  - Learns "M" often completes "ROM"
  - Feed-forward processes

x: [[v1', v2', v3']]  (1, 3, 384) - refined representations
```

**After Block 6**:
```
Position 0 (index 46, "R"):
  - Now knows it's start of a name, likely followed by vowels

Position 1 (index 47, "O"):
  - Knows it's part of "ROM", part of "ROMEO"

Position 2 (index 56, "M"):
  - Knows it completes "ROM", next char likely "E" or "'"

x: [[v1'', v2'', v3'']]  (1, 3, 384) - deep, abstract representations
```

**After output projection**:
```
logits[0, 2] = [score for "a", score for "b", ..., score for "E", ..., score for "z"]
```

Say logits for position 2 (after "M") are:
```
"E": 5.2
"'": 3.1
"e": 2.8
... (all others lower)
```

**After softmax**:
```
"E": 0.72
"'": 0.15
"e": 0.08
... (all others very low)
```

**Sample**: 72% chance pick "E", 15% chance pick "'", etc.

**Output**: "ROME" (most likely)

---

## PART 8: KEY INSIGHTS

1. **Shapes stay consistent**: Most layers output to same shape they receive `(B, T, 384)`
2. **Information gets richer**: While shapes stay same, the content becomes more meaningful
3. **Everything is differentiable**: All operations have gradients, enabling end-to-end learning
4. **Parallel processing**: Attention computes all-to-all relationships in parallel (matrix multiplication)
5. **Hierarchical learning**: Early layers learn simple patterns, later layers learn complex patterns
6. **Residuals enable depth**: Without residuals, training 6+ layers would be extremely difficult

---

## PART 9: HOW ALL PIECES CONNECT

```
┌─────────────────────────────────────────────────────────┐
│                    GPTLanguageModel                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input: idx (B, T)                                      │
│    ↓                                                     │
│  ┌─────────────────────────────────────────────┐       │
│  │ Embeddings (token + position)                │       │
│  │ Output: x (B, T, 384)                        │       │
│  └─────────────────────────────────────────────┘       │
│    ↓                                                     │
│  ┌─────────────────────────────────────────────┐       │
│  │ Block 1                                       │       │
│  │   ├─ MultiHeadAttention (6 heads)            │       │
│  │   │   ├─ Head 1: Q, K, V, attention          │       │
│  │   │   ├─ Head 2: Q, K, V, attention          │       │
│  │   │   └─ ...                                   │       │
│  │   ├─ FeedForward                             │       │
│  │   └─ Residuals + LayerNorms                 │       │
│  │ Output: x (B, T, 384)                        │       │
│  └─────────────────────────────────────────────┘       │
│    ↓                                                     │
│  ┌─────────────────────────────────────────────┐       │
│  │ Block 2 (same structure)                    │       │
│  │ Output: x (B, T, 384)                        │       │
│  └─────────────────────────────────────────────┘       │
│    ↓                                                     │
│  ... (Blocks 3-6)                                        │
│    ↓                                                     │
│  ┌─────────────────────────────────────────────┐       │
│  │ Block 6                                       │       │
│  │ Output: x (B, T, 384)                        │       │
│  └─────────────────────────────────────────────┘       │
│    ↓                                                     │
│  ┌─────────────────────────────────────────────┐       │
│  │ Final LayerNorm                               │       │
│  │ Output: x (B, T, 384)                        │       │
│  └─────────────────────────────────────────────┘       │
│    ↓                                                     │
│  ┌─────────────────────────────────────────────┐       │
│  │ Output Projection (Linear)                   │       │
│  │ Output: logits (B, T, 65)                    │       │
│  └─────────────────────────────────────────────┘       │
│    ↓                                                     │
│  ┌─────────────────────────────────────────────┐       │
│  │ Softmax                                       │       │
│  │ Output: probs (B, T, 65)                     │       │
│  └─────────────────────────────────────────────┘       │
│    ↓                                                     │
│  Output: probs (B, T, 65)                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
