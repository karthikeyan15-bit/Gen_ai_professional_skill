"""
Task 5: Custom Transformer Backpropagation & Multi-Head Gradient Tracking Engine
---------------------------------------------------------------------------------
Objective: Analyze gradient flow and solve vanishing/exploding gradient problems within
attention layers by executing manual backward-pass calculus without autograd.

Required Tech Stack: PyTorch (with Autograd explicitly disabled via requires_grad=False), NumPy
Formula Derivations:
  Forward:
    Q = X @ W_q,  K = X @ W_k,  V = X @ W_v
    S = Q @ K^T / sqrt(d_k)
    P = softmax(S)
    O = P @ V
    L = 0.5 * sum((O - Target)^2)
  Backward (Calculus):
    dL/dO = O - Target
    dL/dV = P^T @ dL/dO
    dL/dP = dL/dO @ V^T
    dL/dS = P * (dL/dP - sum(dL/dP * P, axis=-1, keepdims=True))   [Softmax Jacobian backward]
    dL/dQ = (dL/dS @ K) / sqrt(d_k)
    dL/dK = (dL/dS^T @ Q) / sqrt(d_k)
    dL/dW_q = X^T @ dL/dQ
    dL/dW_k = X^T @ dL/dK
    dL/dW_v = X^T @ dL/dV
"""

import numpy as np
import torch

class ManualAttentionBackwardEngine:
    def __init__(self, embed_dim: int, head_dim: int):
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.scale = 1.0 / np.sqrt(head_dim)

    def forward_and_backward(
        self,
        X: np.ndarray,
        W_q: np.ndarray,
        W_k: np.ndarray,
        W_v: np.ndarray,
        Target: np.ndarray
    ):
        """
        Executes manual forward and backward pass for a single self-attention head.
        """
        N, D = X.shape

        # --- Forward Pass ---
        Q = X @ W_q                 # (N, d_k)
        K = X @ W_k                 # (N, d_k)
        V = X @ W_v                 # (N, d_k)

        S = (Q @ K.T) * self.scale  # (N, N)

        # Numerically stable Softmax
        S_max = np.max(S, axis=-1, keepdims=True)
        exp_S = np.exp(S - S_max)
        P = exp_S / np.sum(exp_S, axis=-1, keepdims=True)  # (N, N)

        O = P @ V                   # (N, d_k)

        # MSE Loss = 0.5 * ||O - Target||^2
        loss = 0.5 * np.sum((O - Target) ** 2)

        # --- Manual Backward Pass Calculus ---
        dL_dO = (O - Target)                                       # (N, d_k)

        # dL/dV = P^T @ dL/dO
        dL_dV = P.T @ dL_dO                                        # (N, d_k)

        # dL/dP = dL/dO @ V^T
        dL_dP = dL_dO @ V.T                                        # (N, N)

        # Softmax backward pass derivative: dL/dS = P * (dL/dP - sum(dL/dP * P, axis=-1))
        sum_dP_P = np.sum(dL_dP * P, axis=-1, keepdims=True)
        dL_dS = P * (dL_dP - sum_dP_P)                            # (N, N)

        # dL/dQ = (dL/dS @ K) * scale
        dL_dQ = (dL_dS @ K) * self.scale                           # (N, d_k)

        # dL/dK = (dL/dS^T @ Q) * scale
        dL_dK = (dL_dS.T @ Q) * self.scale                         # (N, d_k)

        # Gradients wrt Weight Matrices
        dL_dWq = X.T @ dL_dQ                                       # (D, d_k)
        dL_dWk = X.T @ dL_dK                                       # (D, d_k)
        dL_dWv = X.T @ dL_dV                                       # (D, d_k)

        grads = {
            "dW_q": dL_dWq,
            "dW_k": dL_dWk,
            "dW_v": dL_dWv,
            "dQ": dL_dQ,
            "dK": dL_dK,
            "dV": dL_dV
        }

        return loss, grads


def verify_with_pytorch_autograd(X_np, Wq_np, Wk_np, Wv_np, Target_np, engine_grads):
    """
    Compares hand-calculated gradient tensors with PyTorch's automatic differentiation engine.
    """
    X_pt = torch.tensor(X_np, requires_grad=False)
    Wq_pt = torch.tensor(Wq_np, requires_grad=True)
    Wk_pt = torch.tensor(Wk_np, requires_grad=True)
    Wv_pt = torch.tensor(Wv_np, requires_grad=True)
    Target_pt = torch.tensor(Target_np, requires_grad=False)

    head_dim = Wq_np.shape[1]
    scale = 1.0 / (head_dim ** 0.5)

    Q_pt = X_pt @ Wq_pt
    K_pt = X_pt @ Wk_pt
    V_pt = X_pt @ Wv_pt

    S_pt = (Q_pt @ K_pt.T) * scale
    P_pt = torch.softmax(S_pt, dim=-1)
    O_pt = P_pt @ V_pt

    loss_pt = 0.5 * torch.sum((O_pt - Target_pt) ** 2)
    loss_pt.backward()

    # Compare gradients
    err_dWq = np.max(np.abs(engine_grads["dW_q"] - Wq_pt.grad.numpy()))
    err_dWk = np.max(np.abs(engine_grads["dW_k"] - Wk_pt.grad.numpy()))
    err_dWv = np.max(np.abs(engine_grads["dW_v"] - Wv_pt.grad.numpy()))

    print("\n--- Numerical Verification against PyTorch Autograd ---")
    print(f"Max Absolute Gradient Error for dW_q: {err_dWq:.8e}")
    print(f"Max Absolute Gradient Error for dW_k: {err_dWk:.8e}")
    print(f"Max Absolute Gradient Error for dW_v: {err_dWv:.8e}")

    assert err_dWq < 1e-5 and err_dWk < 1e-5 and err_dWv < 1e-5, "Manual backward calculus mismatch!"
    print("Exact mathematical agreement confirmed!")


def track_gradient_flow_across_seq_lengths():
    """
    Simulates gradient flow magnitude behavior across sequence lengths (N=16, 32, 64, 128).
    """
    print("\n--- Tracking Gradient Flow Across Sequence Lengths ---")
    embed_dim = 32
    head_dim = 16
    engine = ManualAttentionBackwardEngine(embed_dim, head_dim)

    np.random.seed(42)
    W_q = np.random.randn(embed_dim, head_dim) * 0.1
    W_k = np.random.randn(embed_dim, head_dim) * 0.1
    W_v = np.random.randn(embed_dim, head_dim) * 0.1

    seq_lengths = [16, 32, 64, 128]
    print(f"{'Seq Length (N)':<15} | {'||dW_q|| Frobenius':<20} | {'||dW_k|| Frobenius':<20} | {'||dW_v|| Frobenius':<20}")
    print("-" * 82)

    for seq_len in seq_lengths:
        X = np.random.randn(seq_len, embed_dim)
        Target = np.random.randn(seq_len, head_dim)

        loss, grads = engine.forward_and_backward(X, W_q, W_k, W_v, Target)
        norm_dWq = np.linalg.norm(grads["dW_q"])
        norm_dWk = np.linalg.norm(grads["dW_k"])
        norm_dWv = np.linalg.norm(grads["dW_v"])

        print(f"{seq_len:<15} | {norm_dWq:<20.4f} | {norm_dWk:<20.4f} | {norm_dWv:<20.4f}")


def main():
    print("=" * 70)
    print("Task 5: Custom Transformer Backpropagation & Gradient Tracking")
    print("=" * 70)

    np.random.seed(42)
    seq_len = 8
    embed_dim = 16
    head_dim = 8

    X = np.random.randn(seq_len, embed_dim)
    W_q = np.random.randn(embed_dim, head_dim)
    W_k = np.random.randn(embed_dim, head_dim)
    W_v = np.random.randn(embed_dim, head_dim)
    Target = np.random.randn(seq_len, head_dim)

    engine = ManualAttentionBackwardEngine(embed_dim, head_dim)
    loss, grads = engine.forward_and_backward(X, W_q, W_k, W_v, Target)

    print(f"Calculated MSE Loss: {loss:.6f}")
    print(f"Gradient Shapes -> dW_q: {grads['dW_q'].shape}, dW_k: {grads['dW_k'].shape}, dW_v: {grads['dW_v'].shape}")

    verify_with_pytorch_autograd(X, W_q, W_k, W_v, Target, grads)
    track_gradient_flow_across_seq_lengths()

    print("\nTask 5 completed successfully!")

if __name__ == "__main__":
    main()
