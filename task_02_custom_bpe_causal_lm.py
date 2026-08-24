"""
Task 2: Custom Byte-Pair Encoding (BPE) Tokenizer & Autoregressive Causal LM
-------------------------------------------------------------------------
Objective: Understand subword tokenization and autoregressive sequence generation
by building a language model training loop from the ground up using PyTorch (tensor ops only).

Required Tech Stack: PyTorch, raw Python, NumPy
"""

from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. Custom Byte-Pair Encoding (BPE) Tokenizer
# =====================================================================

class SimpleBPETokenizer:
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.encoder: dict[str, int] = {}
        self.decoder: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []

    def _get_stats(self, vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: tuple[str, str], v_in: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
        v_out = {}
        bigram = pair
        replacement = "".join(pair)
        for word in v_in:
            w_out = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram[0] and word[i + 1] == bigram[1]:
                    w_out.append(replacement)
                    i += 2
                else:
                    w_out.append(word[i])
                    i += 1
            v_out[tuple(w_out)] = v_in[word]
        return v_out

    def train(self, corpus: list[str]):
        # Word frequency counting with special end-of-word token '</w>'
        words_counts = Counter()
        for text in corpus:
            for word in text.split():
                if word:
                    words_counts[word] += 1

        # Represent words as tuples of characters
        vocab = {tuple(list(word) + ["</w>"]): count for word, count in words_counts.items()}

        # Initialize base alphabet
        alphabet = set()
        for word_tuple in vocab.keys():
            for symbol in word_tuple:
                alphabet.add(symbol)

        num_merges = self.vocab_size - len(alphabet)
        num_merges = max(0, num_merges)

        print(f"[BPE] Initial base vocabulary size: {len(alphabet)}")

        for i in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)

        # Build final encoder / decoder mapping
        all_tokens = set(alphabet)
        for pair in self.merges:
            all_tokens.add("".join(pair))

        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        full_vocab = special_tokens + sorted(list(all_tokens))

        self.encoder = {token: idx for idx, token in enumerate(full_vocab)}
        self.decoder = {idx: token for idx, token in enumerate(full_vocab)}
        print(f"[BPE] Final Vocabulary Size: {len(self.encoder)} with {len(self.merges)} merge rules.")

    def encode(self, text: str) -> list[int]:
        tokens = []
        for word in text.split():
            symbols = list(word) + ["</w>"]
            for pair in self.merges:
                bigram = pair
                replacement = "".join(pair)
                i = 0
                new_symbols = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == bigram[0] and symbols[i + 1] == bigram[1]:
                        new_symbols.append(replacement)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols
            for sym in symbols:
                tokens.append(self.encoder.get(sym, self.encoder.get("<unk>", 1)))
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        raw = "".join([self.decoder.get(idx, "") for idx in token_ids])
        return raw.replace("</w>", " ")


# =====================================================================
# 2. Autoregressive Causal Language Model in PyTorch
# =====================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Custom lower-triangular causal mask to prevent look-ahead leaks
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class CausalLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_heads: int = 4, max_seq_len: int = 64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 10) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -64:] # Crop to context window
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] # Last timestep
            probs = F.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def main():
    print("=" * 70)
    print("Task 2: Custom BPE Tokenizer & Autoregressive Causal LM Verification")
    print("=" * 70)

    # 1. Train Tokenizer
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the generative artificial intelligence transformer model",
        "attention is all you need for autoregressive language modeling"
    ]
    tokenizer = SimpleBPETokenizer(vocab_size=80)
    tokenizer.train(corpus)

    test_sentence = "the transformer model uses attention"
    encoded = tokenizer.encode(test_sentence)
    decoded = tokenizer.decode(encoded)
    print(f"\nSample Text: '{test_sentence}'")
    print(f"Encoded Token IDs: {encoded}")
    print(f"Decoded Text: '{decoded}'")

    # 2. Train Causal LM Loop
    vocab_size = len(tokenizer.encoder)
    model = CausalLM(vocab_size=vocab_size, embed_dim=32, num_heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Prepare dummy dataset batch
    seq_ids = tokenizer.encode("the quick brown fox jumps over the lazy dog")
    if len(seq_ids) < 6:
        seq_ids = seq_ids * 3
    x_input = torch.tensor([seq_ids[:-1]], dtype=torch.long) # Inputs
    y_target = torch.tensor([seq_ids[1:]], dtype=torch.long) # Targets shift by 1

    print("\n--- Training Loop Simulation (10 Steps) ---")
    model.train()
    for step in range(1, 11):
        optimizer.zero_grad()
        logits = model(x_input)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y_target.view(-1))
        loss.backward()
        optimizer.step()
        if step % 2 == 0:
            print(f"Step {step:02d} | Cross-Entropy Loss: {loss.item():.4f}")

    # 3. Autoregressive Generation Test
    model.eval()
    start_tokens = torch.tensor([[encoded[0]]], dtype=torch.long)
    generated_ids = model.generate(start_tokens, max_new_tokens=8)
    gen_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"\nAutoregressively Generated Tokens: {generated_ids.tolist()[0]}")
    print(f"Decoded Output: '{gen_text}'")
    print("\nTask 2 completed successfully!")

if __name__ == "__main__":
    main()
