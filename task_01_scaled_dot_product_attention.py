"""
Task 1: Vectorized Scaled Dot-Product Attention from Scratch
-----------------------------------------------------------
Objective: Master the mathematical foundation of modern Transformer models by
deriving and implementing the core self-attention mechanism using optimized linear
algebra operations.

Required Tech Stack: Pure NumPy, Python 3.10+
Formula: Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) ) @ V
"""

import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable Softmax calculation over specified axis.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized 4D Scaled Dot-Product Attention.

    Args:
        query: Tensor of shape (batch_size, num_heads, seq_len_q, head_dim)
        key:   Tensor of shape (batch_size, num_heads, seq_len_k, head_dim)
        value: Tensor of shape (batch_size, num_heads, seq_len_v, head_dim) (seq_len_k == seq_len_v)
        mask:  Optional mask tensor broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k)
               1 or True indicates allowed tokens; 0 or False indicates masked tokens.

    Returns:
        output:           Tensor of shape (batch_size, num_heads, seq_len_q, head_dim)
        attention_weights: Tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    # 1. Verify dimensional compatibility
    assert query.ndim == 4, f"Query must be 4D (batch, heads, seq_len, head_dim), got shape {query.shape}"
    assert key.ndim == 4, f"Key must be 4D (batch, heads, seq_len, head_dim), got shape {key.shape}"
    assert value.ndim == 4, f"Value must be 4D (batch, heads, seq_len, head_dim), got shape {value.shape}"

    head_dim = query.shape[-1]
    scale_factor = np.sqrt(head_dim)

    # 2. Compute Raw Scaled Scores: Q @ K^T / sqrt(d_k)
    # Transpose last two dimensions of Key: (batch, heads, head_dim, seq_len_k)
    key_transposed = np.swapaxes(key, -1, -2)
    scores = np.matmul(query, key_transposed) / scale_factor

    # 3. Apply Causal / Padding Mask if provided
    if mask is not None:
        # Fill masked positions with negative infinity so softmax maps them to 0
        scores = np.where(mask == 0, -1e9, scores)

    # 4. Softmax over key sequence dimension
    attention_weights = softmax(scores, axis=-1)

    # 5. Compute Weighted Sum of Values: Attention_Weights @ V
    output = np.matmul(attention_weights, value)

    return output, attention_weights

def create_casual_mask(seq_len: int) -> np.ndarray:
    """
    Creates a lower-triangular causal mask for decoder simulation.
    1 for allowed past/present positions, 0 for future masked positions.
    """
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.int32))
    return mask

def main():
    print("=" * 70)
    print("Task 1: Vectorized Scaled Dot-Product Attention Verification")
    print("=" * 70)

    # Hyperparameters
    batch_size = 2
    num_heads = 4
    seq_len = 5
    head_dim = 16

    np.random.seed(42)

    # Generate synthetic input tensors [batch, num_heads, seq_len, head_dim]
    Q = np.random.randn(batch_size, num_heads, seq_len, head_dim)
    K = np.random.randn(batch_size, num_heads, seq_len, head_dim)
    V = np.random.randn(batch_size, num_heads, seq_len, head_dim)

    print(f"Input Shapes -> Q: {Q.shape}, K: {K.shape}, V: {V.shape}")

    # Test 1: Unmasked Multi-Head Attention
    out_unmasked, attn_weights_unmasked = scaled_dot_product_attention(Q, K, V)
    print(f"\n[1] Unmasked Output Shape: {out_unmasked.shape}")
    print(f"    Attention Weights Shape: {attn_weights_unmasked.shape}")
    print(f"    Attention Probabilities Sum across seq (head 0, batch 0): {attn_weights_unmasked[0, 0].sum(axis=-1)}")

    # Test 2: Causal Masked Multi-Head Attention
    causal_mask = create_casual_mask(seq_len)
    out_masked, attn_weights_masked = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    print(f"\n[2] Masked Output Shape: {out_masked.shape}")
    print(f"    Masked Attention Weights (Batch 0, Head 0):\n{attn_weights_masked[0, 0].round(3)}")

    # Assert upper triangular (future tokens) attention probabilities are exactly zero
    upper_tri = np.triu(attn_weights_masked[0, 0], k=1)
    assert np.allclose(upper_tri, 0.0), "Causal mask failed! Upper triangular values are non-zero."

    print("\nTask 1 completed successfully! Attention math verified.")

if __name__ == "__main__":
    main()
