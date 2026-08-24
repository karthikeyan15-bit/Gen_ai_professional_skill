"""
Task 3: Local 7B Parameter LLM Quantization & Logit Extraction Pipeline
-----------------------------------------------------------------------
Objective: Leverage cutting-edge model compression techniques to run large language
models on edge devices, while building real-time interpretability hooks to inspect layer-wise
logits and attention weights, calculating dynamic entropy bounds across generation steps.

Required Tech Stack: PyTorch, NumPy
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. 4-bit Integer Quantization & Dequantization (Q4_K Block Simulation)
# =====================================================================

class QuantizerINT4:
    """
    Simulates 4-bit Uniform Quantization (Q4_K_M representation)
    Range for signed 4-bit integer: [-8, 7]
    """
    @staticmethod
    def quantize_tensor(tensor: torch.Tensor, qmin: int = -8, qmax: int = 7) -> tuple[torch.Tensor, float, float]:
        min_val = tensor.min().item()
        max_val = tensor.max().item()

        # Prevent division by zero
        if max_val == min_val:
            scale = 1.0
            zero_point = 0.0
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - (min_val / scale)

        q_tensor = torch.round(tensor / scale + zero_point)
        q_tensor = torch.clamp(q_tensor, qmin, qmax).to(torch.int8)

        return q_tensor, scale, zero_point

    @staticmethod
    def dequantize_tensor(q_tensor: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
        return (q_tensor.to(torch.float32) - zero_point) * scale


# =====================================================================
# 2. Analytical Extraction Hooks & Model Inspection Pipeline
# =====================================================================

class AnalyticalExtractionModel(nn.Module):
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, vocab_size)

        # Hook storage dictionary
        self.layer_activations: dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        for idx, layer in enumerate(self.layers):
            layer_name = f"layer_{idx}"
            layer.register_forward_hook(self._make_hook(layer_name))

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self.layer_activations[name] = output.detach()
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.head(x)
        return logits


def calculate_shannon_entropy(logits: torch.Tensor) -> float:
    """
    Computes Shannon Entropy H(X) = - sum( p(x) * log2(p(x)) ) over logit probabilities.
    """
    probs = F.softmax(logits, dim=-1)
    # Avoid log2(0) by adding small epsilon
    log_probs = torch.log2(probs + 1e-12)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean().item()


def main():
    print("=" * 70)
    print("Task 3: Quantization & Analytical Logit Extraction Verification")
    print("=" * 70)

    # -------------------------------------------------------------
    # Part 1: Quantization & Dequantization Error Analysis
    # -------------------------------------------------------------
    print("\n--- [Part 1] INT4 Quantization Simulation ---")
    torch.manual_seed(42)
    weight_tensor = torch.randn(64, 64, dtype=torch.float32)

    q_weight, scale, zero_pt = QuantizerINT4.quantize_tensor(weight_tensor)
    deq_weight = QuantizerINT4.dequantize_tensor(q_weight, scale, zero_pt)

    mse_loss = F.mse_loss(deq_weight, weight_tensor).item()
    compression_ratio = (weight_tensor.element_size() * 8) / 4  # FP32 (32-bit) vs INT4 (4-bit)

    print(f"Original Weight Shape: {weight_tensor.shape}")
    print(f"Quantized INT4 Value Min/Max: [{q_weight.min().item()}, {q_weight.max().item()}]")
    print(f"Scale: {scale:.6f} | Zero-Point: {zero_pt:.2f}")
    print(f"Reconstruction Error (MSE): {mse_loss:.6f}")
    print(f"Theoretical Compression Speedup / Storage Reduction: {compression_ratio:.1f}x")

    # -------------------------------------------------------------
    # Part 2: Layer-Wise Logit Extraction & Dynamic Entropy Bounds
    # -------------------------------------------------------------
    print("\n--- [Part 2] Hidden Layer Hooks & Entropy Bounds Extraction ---")
    vocab_size = 500
    model = AnalyticalExtractionModel(vocab_size=vocab_size, hidden_dim=64, num_layers=4)
    model.eval()

    # Input sequence
    input_tokens = torch.randint(0, vocab_size, (1, 8))

    with torch.no_grad():
        logits = model(input_tokens)

    print(f"Extracted activations from {len(model.layer_activations)} hidden layers:")
    for name, act in model.layer_activations.items():
        act_norm = torch.norm(act, p=2).item()
        print(f"  -> Layer '{name}' activation tensor shape: {act.shape} | L2 Norm: {act_norm:.4f}")

    # Dynamic Entropy calculation across 5 generation steps
    print("\n--- Autoregressive Step-Wise Dynamic Entropy Bounds ---")
    curr_tokens = input_tokens
    for step in range(1, 6):
        with torch.no_grad():
            out_logits = model(curr_tokens)
            last_token_logits = out_logits[:, -1, :]
            entropy = calculate_shannon_entropy(last_token_logits)

            # Sample next token greedily
            next_token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

            print(f"Generation Step {step:02d} | Next Token ID: {next_token.item():4d} | Logit Shannon Entropy: {entropy:.4f} bits")

    print("\nTask 3 completed successfully!")

if __name__ == "__main__":
    main()
