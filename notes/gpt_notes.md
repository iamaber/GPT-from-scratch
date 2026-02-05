## 1. Data Loading & Exploration

### What we're doing

- Reading Shakespeare's text from `input.txt`
- The dataset contains 1,115,394 characters

### Why this matters

Language models need text data to learn patterns. Shakespeare's works are great because they have rich vocabulary and consistent grammar.

---

## 2. Vocabulary Building

### What we're doing

```python
chars = sorted(list(set(text)))  # Get unique characters
vocab_size = len(chars)          # 65 unique characters
```

### The vocabulary (65 characters)

Includes: lowercase letters (a-z), uppercase (A-Z), punctuation (! ? . , ; :), numbers, and special characters like newline (\n) and space.

### Why character-level tokenization

- Simple and interpretable
- No unknown tokens (every character is in vocabulary)
- Trade-off: longer sequences than word/token-level models

---

## 3. Encoding & Decoding

### What we're doing

```python
stoi = {ch: i for i, ch in enumerate(chars)}  # string to integer
itos = {i: ch for i, ch in enumerate(chars)}  # integer to string

encode = lambda s: [stoi[c] for c in s]   # "hello" → [45, 40, 50, 50, 55]
decode = lambda l: ''.join([itos[i] for i in l])  # [45, 40, 50, 50, 55] → "hello"

```

### Example

```
"hi, i am aber" → [46, 47, 6, 1, 47, 1, 39, 51, 1, 39, 40, 43, 56]

```

### Why we need this

- Neural networks work with numbers, not text
- Encoding converts text to model input
- Decoding converts model output back to readable text

---

## 4. Converting to PyTorch Tensors

### What we're doing

```python
data = torch.tensor(encode(text), dtype=torch.long)

```

### Why `dtype=torch.long` (64-bit integers)

- Embedding layers expect integer indices
- `long` type is the standard for indices in PyTorch
- Avoids precision issues with float types

### Tensor shape

```
torch.Size([1115394])  # 1D tensor, one number per character

```

---

## 5. Train/Validation Split

### What we're doing

```python
n = int(0.9 * len(data))  # 90% training, 10% validation
train_data = data[:n]
val_data = data[n:]

```

### Why split the data

- **Training set**: Teach the model patterns (90% of data)
- **Validation set**: Check if model overfits (10% of data)
- Model should perform similarly on both sets

### Overfitting

- If train loss is low but val loss is high → model memorized training data
- We want both losses to decrease together

---

## 6. Context Window (`block_size`)

### What `block_size` means

```python
block_size = 256  # Maximum context length (GPT model)
```

This is how many previous characters the model can "see" when predicting the next one.

### How it works: Input → Target pairs

| Input (context) | Target (next character) |
| --- | --- |
| `[18]` | `47` |
| `[18, 47]` | `56` |
| `[18, 47, 56]` | `57` |
| `[18, 47, 56, 57, 58, 1, 15, 47]` | `58` |

### Why multiple context lengths

- Model learns to use varying amounts of context
- Short contexts (single char) for immediate patterns
- Long contexts for grammar and structure
- Makes training data more efficient (8 examples from 9 characters)

---

## 7. Batching (`get_batch` function)

### What batching does

```python
batch_size = 64   # Process 64 sequences at once (GPT model)
block_size = 256  # Each sequence is 256 characters long (GPT model)
```

### Random sampling strategy

```python
ix = torch.randint(len(data) - block_size, (batch_size,))

```

- Pick `batch_size` random starting positions
- Extract `block_size` characters from each position
- Create targets by shifting one position forward

### Output shapes

```
x (inputs):  torch.Size([64, 256])  # 64 sequences, 256 characters each (GPT model)
y (targets): torch.Size([64, 256])  # 64 sequences, 256 targets each

```

### Why batch

- **Parallel processing**: GPU processes multiple sequences simultaneously
- **Faster training**: More data per forward/backward pass
- **Better gradient estimates**: Average over multiple examples

---

## 8. Bigram Language Model

### Model architecture

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```

### How embeddings work

- **Embedding table**: 65 rows × 65 columns (vocab_size × vocab_size)
- Each row represents the "logits" (unnormalized probabilities) for the next character
- Given character `i`, the model looks up row `i` to predict what comes next

### Forward pass shapes

```
idx (input):  (B, T) → (4, 8)         # batch, time
logits:       (B, T, C) → (4, 8, 65)  # batch, time, channels (vocab)

```

### Loss calculation

```python
loss = F.cross_entropy(logits, targets)

```

- Reshapes logits from `(B, T, C)` to `(B*T, C)` - flattens batch and time
- Reshapes targets from `(B, T)` to `(B*T)` - flattens batch and time
- Computes cross-entropy between predicted and actual characters

### Understanding loss

- Lower loss = better predictions
- Random model: ~4.18 (log(65) because 65 possible characters)
- Trained model: Should be much lower

### Generation loop

```python
for _ in range(max_new_tokens):
    logits, loss = self(idx)
    logits = logits[:, -1, :]        # Take last time step only
    probs = F.softmax(logits, dim=-1)  # Convert to probabilities
    idx_next = torch.multinomial(probs, num_samples=1)  # Sample from distribution
    idx = torch.cat((idx, idx_next), dim=1)  # Append to sequence
```

### Why sample instead of taking max

- **Max (greedy)**: Always picks most likely character → repetitive text
- **Sampling**: Picks based on probability distribution → more variety and natural text
- Example: If 'e' has 0.3 probability, 't' has 0.2, model might pick either

---

## 9. Optimization & Training

### Optimizer choice: AdamW

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

```

### Why AdamW

- **Adaptive learning rate**: Adjusts per parameter based on gradient history
- **Weight decay**: Regularization to prevent overfitting
- Works well out-of-the-box for most models

### Learning rate (lr)

```python
lr = 1e-2  # 0.01

```

- **Too high**: Training becomes unstable, loss may explode
- **Too low**: Training is very slow
- **Just right**: Smooth decrease in loss

### Training loop steps

```python
for iter in range(max_iters):
    # 1. Get batch
    xb, yb = get_batch('train')

    # 2. Forward pass (compute loss)
    logits, loss = model(xb, yb)

    # 3. Zero gradients (clear previous step)
    optimizer.zero_grad()

    # 4. Backward pass (compute gradients)
    loss.backward()

    # 5. Update parameters
    optimizer.step()
```

### Evaluation during training

```python
if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}")

```

### Why evaluate periodically

- Monitor training progress
- Detect overfitting (val loss increasing while train loss decreases)
- Decide when to stop training

---

## 10. Self-Attention: The Mathematical Trick

### Goal: Weighted aggregation

We want each position to see information from all previous positions:

```
x[b,t] = mean(x[b,0], x[b,1], ..., x[b,t])

```

### Version 1: For loops (slow)

```python
for b in range(B):
    for t in range(T):
        xbow[b,t] = torch.mean(x[b,:t+1], 0)

```

### Version 2: Matrix multiplication (fast)

```python
wei = torch.tril(torch.ones(T, T))      # Lower triangular matrix
wei = wei / wei.sum(1, keepdim=True)   # Normalize rows
xbow = wei @ x                          # Matrix multiply

```

### How it works

```
wei (weights)         x (data)
[1.0  0.0  0.0]       [2.0  7.0]
[0.5  0.5  0.0]   @   [6.0  4.0]   =   [weighted averages]
[0.33 0.33 0.33]      [6.0  5.0]

```

- **Lower triangular**: Each position only sees itself and previous positions
- **Row sums to 1**: Equal weighting of visible positions
- **Matrix multiply**: Efficiently computes weighted sums

### Version 3: Softmax (what transformers use)

```python
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))  # Mask future
wei = F.softmax(wei, dim=-1)                    # Convert to probabilities
xbow = wei @ x

```

### Why softmax matters

- **Attention mechanism**: Learned weights instead of equal averaging
- **`inf` mask**: Ensures positions can't see the future
- **Softmax**: Converts numbers to probabilities that sum to 1

### Connection to transformers

- In real transformers, `wei` is computed from queries and keys
- This example shows the core idea: weighted aggregation of past information
- Self-attention allows each position to "pay attention" to different previous positions

---

## 10.5. Positional Embeddings

### Why we need them

Token embeddings alone don't know where a token appears in the sequence. "cat" at position 1 and "cat" at position 100 would have the same embedding.

### How they work

```python
class GPTLanguageModel(nn.Module):
    def __init__(self):
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
```

### Forward pass

```python
tok_emb = self.token_embedding_table(idx)      # (B,T,C) - token identities
pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C) - positions
x = tok_emb + pos_emb  # (B,T,C) - combine both
```

### Shapes

- `tok_emb`: (batch_size=64, block_size=256, n_embd=384)
- `pos_emb`: (block_size=256, n_embd=384)
- `x`: (64, 256, 384) - broadcasted addition

### Why this works

- Addition preserves both identity and position information
- Learned embeddings: the model learns optimal position representations
- Maximum position = block_size (256), can't generate longer sequences without recomputing

---

## 11. Single Attention Head

### The core idea

Each position in the sequence should selectively pay attention to different previous positions.

### Query, Key, Value (Q, K, V)

```python
class Head(nn.Module):
    def __init__(self, head_size):
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
```

- **Query (Q)**: "What am I looking for?" (current position)
- **Key (K)**: "What do I contain?" (all positions)
- **Value (V)**: "What information do I share?" (all positions)

### Attention mechanism

```python
k = self.key(x)      # (B, T, head_size) - keys for all positions
q = self.query(x)    # (B, T, head_size) - queries for all positions
wei = q @ k.transpose(-2, -1) * head_size**-0.5  # (B, T, T) - attention scores
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Mask future
wei = F.softmax(wei, dim=-1)  # (B, T, T) - attention weights
v = self.value(x)   # (B, T, head_size) - values for all positions
out = wei @ v       # (B, T, head_size) - weighted aggregation
```

### Scaled dot-product attention

```python
wei = q @ k.transpose(-2, -1) * head_size**-0.5
```

- **Without scaling**: Large dot products → softmax becomes extremely peaked (0s and 1s)
- **With scaling** (`1/sqrt(head_size)`): Stable gradients, smoother attention distribution
- Why `sqrt(head_size)`: Variance of dot product of two unit vectors = dimension

### Masking future

```python
tril = torch.tril(torch.ones(block_size, block_size))
wei = wei.masked_fill(tril[:T, :T] == 0, float('-inf'))
```

- Lower triangular mask: Position t can only see positions 0 to t
- `float('-inf')`: After softmax, these become 0 (no attention to future)
- Why needed: During training, we use teacher forcing (know the target), but model shouldn't cheat

### Shapes in attention

| Operation | Shape | Description |
| --- | --- | --- |
| `x` (input) | (B=64, T=256, C=384) | Input embeddings |
| `k` | (64, 256, 64) | Keys (head_size = n_embd/n_head = 384/6 = 64) |
| `q` | (64, 256, 64) | Queries |
| `q @ k.T` | (64, 256, 256) | Attention scores (all-to-all) |
| `wei` (after softmax) | (64, 256, 256) | Attention weights |
| `v` | (64, 256, 64) | Values |
| `out` | (64, 256, 64) | Weighted values (output of head) |

### Dropout in attention

```python
self.dropout = nn.Dropout(dropout)  # dropout = 0.2
wei = self.dropout(wei)
```

- Applied after softmax to attention weights
- Prevents over-reliance on specific positions
- Training only (not during generation)

---

## 12. Multi-Head Attention

### Why multiple heads?

Single head learns one type of attention pattern. Multiple heads can learn different patterns in parallel:
- Head 1: May focus on grammatical structure
- Head 2: May focus on semantic relationships
- Head 3: May focus on long-range dependencies
- And so on...

### Architecture

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
```

### Parameters

```python
n_embd = 384    # Embedding dimension
n_head = 6      # Number of attention heads
head_size = n_embd // n_head  # 384 // 6 = 64
```

### Forward pass

```python
def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate heads
    out = self.dropout(self.proj(out))  # Project back to n_embd
    return out
```

### Shapes

| Operation | Shape | Description |
| --- | --- | --- |
| `x` (input) | (64, 256, 384) | Input embeddings |
| `h(x)` (one head) | (64, 256, 64) | Output of one attention head |
| `torch.cat(..., dim=-1)` | (64, 256, 384) | Concatenate 6 heads (6 × 64 = 384) |
| `proj(out)` | (64, 256, 384) | Linear projection (optional, learns to mix heads) |

### Why projection layer?

- Concatenation alone just stacks head outputs
- Projection allows heads to share information
- Model can learn how to best combine information from all heads

---

## 13. Feed-Forward Networks

### Purpose

Attention handles communication between positions. Feed-forward handles computation/processing of each position independently.

### Architecture

```python
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 384 → 1536 (4x expansion)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # 1536 → 384 (back to original)
            nn.Dropout(dropout),
        )
```

### Why 4x expansion?

- Gives model more capacity for non-linear transformations
- Common transformer design choice (from original Transformer paper)
- Balance between compute and model capacity

### Shapes

| Operation | Shape | Description |
| --- | --- | --- |
| `x` (input) | (64, 256, 384) | Output of attention |
| `Linear(384 → 1536)` | (64, 256, 1536) | Expand dimensions |
| `ReLU` | (64, 256, 1536) | Non-linear activation |
| `Linear(1536 → 384)` | (64, 256, 384) | Project back |
| `Dropout(0.2)` | (64, 256, 384) | Regularization |

### Why needed after attention?

- Attention is linear (weighted sum of values)
- Feed-forward adds non-linearity (ReLU)
- Allows model to learn complex transformations beyond weighted averaging

---

## 14. Layer Normalization

### Purpose

Stabilize training by normalizing activations across features (not across batch like BatchNorm).

### Architecture

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
```

### Pre-norm vs Post-norm

```python
def forward(self, x):
    x = x + self.sa(self.ln1(x))  # Pre-norm: Normalize BEFORE attention
    x = x + self.ffwd(self.ln2(x))  # Pre-norm: Normalize BEFORE feed-forward
    return x
```

**Why Pre-norm?**
- More stable training for deep networks
- Gradients flow better through residual connections
- Modern transformers (GPT-2, GPT-3, etc.) use pre-norm

### How LayerNorm works

```python
normalized = (x - mean) / sqrt(variance + epsilon)
output = gamma * normalized + beta
```

- Normalizes to zero mean, unit variance per sample (not per batch)
- `gamma` and `beta` are learnable parameters (scale and shift)
- `epsilon` prevents division by zero

### Residual connections

```python
x = x + self.sa(self.ln1(x))  # Residual: add input to output
x = x + self.ffwd(self.ln2(x))
```

**Why residuals?**
- Allow gradients to flow unchanged through deep networks
- Model can learn identity function (skip the layer) if needed
- Enables training very deep models (6 layers in our GPT, up to 96 in GPT-3)

---

## 15. Transformer Blocks

### Purpose

Combine communication (attention) and computation (feed-forward) into a reusable unit.

### Architecture

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)  # Communication
        self.ffwd = FeedFoward(n_embd)                   # Computation
        self.ln1 = nn.LayerNorm(n_embd)                   # Normalize
        self.ln2 = nn.LayerNorm(n_embd)                   # Normalize
```

### Forward pass

```python
def forward(self, x):
    # 1. Self-attention with residual connection
    x = x + self.sa(self.ln1(x))
    
    # 2. Feed-forward with residual connection
    x = x + self.ffwd(self.ln2(x))
    
    return x
```

### What happens in one block

| Step | Operation | Purpose |
| --- | --- | --- |
| 1 | `ln1(x)` | Normalize input |
| 2 | `sa(...)` | Self-attention: gather info from other positions |
| 3 | `x + ...` | Residual: preserve original information |
| 4 | `ln2(x)` | Normalize again |
| 5 | `ffwd(...)` | Feed-forward: process gathered info |
| 6 | `x + ...` | Residual: preserve original information |

### Stacking blocks

```python
self.blocks = nn.Sequential(
    *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]  # n_layer = 6
)
```

- Each block learns increasingly complex patterns
- Deeper = more capacity to learn hierarchical features
- Trade-off: More parameters, slower training, risk of overfitting

---

## 16. GPT Architecture

### Complete model

```python
class GPTLanguageModel(nn.Module):
    def __init__(self):
        # Embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        
        # Final layer norm and projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
```

### Forward pass

```python
def forward(self, idx, targets=None):
    B, T = idx.shape
    
    # 1. Embeddings (token + position)
    tok_emb = self.token_embedding_table(idx)  # (B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
    x = tok_emb + pos_emb  # (B,T,C)
    
    # 2. Transformer blocks (attention + feed-forward)
    x = self.blocks(x)  # (B,T,C)
    
    # 3. Final normalization
    x = self.ln_f(x)  # (B,T,C)
    
    # 4. Output projection (logits)
    logits = self.lm_head(x)  # (B,T,vocab_size)
    
    # 5. Loss calculation (if training)
    if targets is not None:
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
    
    return logits, loss
```

### Weight initialization

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

**Why careful initialization?**
- Prevents exploding/vanishing gradients at start of training
- Standard deviations of 0.02 work well for transformer models
- Original Transformer paper and GPT papers use similar values

### Generation

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        # Crop to block_size if needed
        idx_cond = idx[:, -block_size:]
        
        # Get predictions
        logits, loss = self(idx_cond)
        
        # Take last time step
        logits = logits[:, -1, :]  # (B, C)
        
        # Convert to probabilities and sample
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
```

### Total parameter count

```
~10.5M parameters
```

Breakdown:
- Token embeddings: 65 × 384 = 24,960
- Position embeddings: 256 × 384 = 98,304
- 6 Transformer blocks: ~10M parameters (attention + feed-forward)
- Final projection: 384 × 65 = 24,960

---

## 17. Updated Hyperparameters

### Comparison: Bigram vs GPT

| Parameter | Bigram Model | GPT Model | Why changed? |
| --- | --- | --- | --- |
| `batch_size` | 32 | 64 | More samples per step, faster training |
| `block_size` | 8 | 256 | Much longer context for better predictions |
| `max_iters` | 3000 | 5000 | More iterations for larger model |
| `eval_interval` | 300 | 500 | Evaluate less often (longer training) |
| `learning_rate` | 1e-2 (0.01) | 3e-4 (0.0003) | Lower LR for larger model stability |
| `n_embd` | N/A | 384 | Embedding dimension |
| `n_head` | N/A | 6 | Number of attention heads |
| `n_layer` | N/A | 6 | Number of transformer blocks |
| `dropout` | N/A | 0.2 | Regularization to prevent overfitting |

### Model size

| Model | Parameters | Trainable params |
| --- | --- | --- |
| Bigram | ~4,225 (65 × 65) | 4,225 |
| GPT | ~10,500,000 | 10,500,000 |
| **Increase** | **~2,488×** | **~2,488×** |

### Training speed considerations

- **Bigram**: Trains in seconds to minutes
- **GPT**: Trains in minutes to hours (depending on GPU)
- Larger batch size (64) helps GPU utilization
- More parameters → more compute per iteration

---

## 18. Key Concepts Summary

| Concept | Purpose |
| --- | --- |
| **Tokenization** | Convert text to numbers |
| **Embeddings** | Learn representations for each token |
| **Positional embeddings** | Encode token positions in sequence |
| **Context window** | How much history the model considers |
| **Batching** | Parallel processing for speed |
| **Loss** | Measure of prediction error |
| **Optimizer** | Updates model to reduce loss |
| **Attention (Q, K, V)** | Learn which previous tokens are important |
| **Multi-head attention** | Multiple attention patterns in parallel |
| **Feed-forward** | Non-linear transformations per position |
| **Layer normalization** | Stabilize training |
| **Residual connections** | Enable deep networks |
| **Transformer blocks** | Combine attention + feed-forward |

---

## 19. Training Tips

### If loss is not decreasing (GPT model)

- Try different learning rate (1e-3 to 1e-4, currently 3e-4)
- Increase `max_iters` (currently 5000)
- Check GPU availability: `device = "cuda" if torch.cuda.is_available() else "cpu"`
- Verify data loading (check `get_batch` function)

### If generated text is gibberish

- Train longer (increase `max_iters`)
- GPT model should produce coherent text after sufficient training
- Lower dropout if underfitting (currently 0.2)
- Check `block_size` is appropriate (currently 256)

### Monitoring overfitting (GPT model)

```
Good: Both train and val loss decreasing together
Bad: Train loss decreasing, val loss increasing → overfitting

Example GPT training:
step 0:   train loss 4.5472, val loss 4.5530
step 500: train loss 2.1234, val loss 2.1345
step 1000: train loss 1.5678, val loss 1.6012  ← Should track closely
```

### GPT-specific tips

- **Warm-up**: Start with lower learning rate, increase to target
- **Gradient clipping**: Prevent exploding gradients (`torch.nn.utils.clip_grad_norm_`)
- **Learning rate scheduling**: Decrease LR during training
- **Batch size**: Larger batches = more stable gradients, but more memory
- **Mixed precision**: Use `torch.cuda.amp` for faster training on modern GPUs

### Common training issues

| Issue | Symptom | Fix |
| --- | --- | --- |
| Exploding loss | Loss becomes NaN | Lower learning rate, add gradient clipping |
| Not learning | Loss stays flat | Check data, increase LR, verify model architecture |
| Overfitting | Train loss << val loss | Increase dropout, add more data, early stopping |
| Too slow | Training takes forever | Increase batch size, use GPU, mixed precision |

```

---

## 20. What We've Built: From Bigram to GPT

### Starting point: Bigram model

**Features:**
- Simple lookup table for next character prediction
- No context beyond previous character
- ~4K parameters
- Trains in seconds
- Generates gibberish (expected)

**Limitation:** Can only learn immediate character dependencies.

### Ending point: Full GPT model

**Features:**
- Self-attention: Learns to attend to relevant past context
- Multi-head attention: Multiple attention patterns in parallel
- Feed-forward networks: Non-linear transformations
- Layer normalization: Stable training
- Residual connections: Enables deep architecture (6 layers)
- Positional embeddings: Knows token positions
- ~10.5M parameters
- Trains in minutes to hours
- Generates coherent Shakespeare-like text

**Improvement over bigram:** Can learn long-range dependencies, grammar, and style.

### Progression summary

| Component | Bigram | GPT | Improvement |
| --- | --- | --- | --- |
| Context | 1 character | 256 characters | 256× more context |
| Parameters | 4,225 | 10,500,000 | 2,488× more capacity |
| Training time | Seconds | Minutes/Hours | Worth it for quality |
| Output quality | Random | Coherent | Actually readable! |

### What the GPT model learned

After training on Shakespeare:
- **Grammar**: Proper sentence structure, punctuation
- **Vocabulary**: Shakespeare's extensive word choices
- **Style**: Dramatic, poetic language patterns
- **Context**: References characters, themes, and dialogue patterns
- **Structure**: Plays have acts, scenes, speaker names

### Example output comparison

**Bigram (random):**
```
hJkLmN;oPqRsTuVwXyZ!?
```

**GPT (trained):**
```
ROMEO:
What's in a name? That which we call a rose
By any other name would smell as sweet.
```

---

## 21. Common Mistakes to Avoid (GPT Edition)

### Tensor shape errors

- **Wrong shapes**: Pay attention to `(B, T, C)` vs `(B*T, C)` 
- **Attention shape mismatch**: Q, K, V must have compatible dimensions
- **Concatenation dim**: `torch.cat` needs correct dimension (`dim=-1` for heads)

### Off-by-one errors

- **Input/target alignment**: Check `x` and `y` are properly offset
- **Positional embeddings**: Ensure `torch.arange(T)` matches sequence length
- **Masking**: Verify `self.tril[:T, :T]` handles variable lengths correctly

### Device issues

- **Not setting device**: GPU vs CPU can cause massive slowdown (100× slower on CPU)
- **Mixed devices**: All tensors and model must be on same device
- **CUDA out of memory**: Reduce `batch_size` or `n_embd` if GPU runs out of memory

### Memory issues

- **Forgetting `torch.no_grad()`**: During evaluation to save memory
- **Gradient accumulation**: Accumulate gradients over multiple batches if GPU limited
- **Clearing cache**: Use `torch.cuda.empty_cache()` if needed (rarely necessary)

### Training issues

- **Wrong dtype**: Embeddings need integer indices (`long` type)
- **Not zeroing gradients**: Forgetting `optimizer.zero_grad()` before each step
- **Wrong loss computation**: Cross-entropy expects flattened shapes `(B*T, C)`
- **Learning rate**: Too high for GPT (use 1e-4 to 3e-4, not 1e-2 like bigram)

### Architecture mistakes

- **Head size mismatch**: `head_size = n_embd // n_head` must be integer
- **Projection layer**: Forgetting projection in MultiHeadAttention
- **Layer norm position**: Use pre-norm (normalize before, not after)
- **Residual connections**: Don't forget `x = x + ...` after each sub-layer

### Common bugs in implementation

```python
# ❌ Wrong: Q and K transpose order
wei = k @ q.transpose(-2, -1)  # Wrong order!

# ✓ Correct: Q @ K.T
wei = q @ k.transpose(-2, -1)  # Right order

# ❌ Wrong: Forget masking
wei = F.softmax(wei, dim=-1)  # Can see future!

# ✓ Correct: Apply mask first
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

# ❌ Wrong: No projection after concat
out = torch.cat([h(x) for h in self.heads], dim=-1)

# ✓ Correct: Add projection
out = torch.cat([h(x) for h in self.heads], dim=-1)
out = self.proj(out)
```

---

## Resources

- Original implementation: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Attention paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Andrej Karpathy's "Let's build GPT" video series
