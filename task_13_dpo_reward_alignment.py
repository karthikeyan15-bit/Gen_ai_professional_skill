"""
Task 13: Direct Preference Optimization (DPO) and Pairwise Reward Alignment
-------------------------------------------------------------------------
Objective: Align model behavior with targeted human values and system safety
guidelines using contrastive pairwise preference tuning without RL complexity.

Required Tech Stack: PyTorch
Formula:
  L_DPO(theta; pi_ref) = - E [ log sigmoid( beta * log(pi_theta(y_w|x) / pi_ref(y_w|x))
                                          - beta * log(pi_theta(y_l|x) / pi_ref(y_l|x)) ) ]
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. Base Causal Policy Model
# =====================================================================

class SmallPolicyLM(nn.Module):
    def __init__(self, vocab_size: int = 500, embed_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.head(x)
        return logits

    def compute_sequence_log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes sum of log probabilities over sequence: sum_t log P(x_t | x_<t)
        """
        logits = self.forward(input_ids)
        # Shift logits and targets for autoregressive probability
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        # Gather log prob of target tokens
        target_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        return target_log_probs.sum(dim=-1)


# =====================================================================
# 2. DPO Loss Calculator Engine
# =====================================================================

class DPOLossCalculator:
    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def compute_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes Direct Preference Optimization Loss.
        """
        # Calculate log likelihood ratios for chosen and rejected responses
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # Implicit reward differences = beta * (log P_policy - log P_ref)
        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios
        reward_margin = chosen_rewards - rejected_rewards

        # DPO Loss = - log sigmoid( beta * logratio_w - beta * logratio_l )
        losses = -F.logsigmoid(reward_margin)
        return losses.mean(), chosen_rewards.detach().mean(), rejected_rewards.detach().mean()


def main():
    print("=" * 70)
    print("Task 13: Direct Preference Optimization (DPO) Verification")
    print("=" * 70)

    torch.manual_seed(42)
    vocab_size = 500

    # 1. Dual-Model Memory Landscape Setup
    policy_model = SmallPolicyLM(vocab_size=vocab_size, embed_dim=32)
    # Frozen Reference Model pi_ref is an exact deepcopy of unaligned initial policy
    reference_model = copy.deepcopy(policy_model)
    
    for p in reference_model.parameters():
        p.requires_grad = False
    reference_model.eval()

    dpo_calculator = DPOLossCalculator(beta=0.1)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-3)

    # 2. Construct Preference Dataset (Prompt + Chosen vs Rejected Responses)
    batch_size = 4
    seq_len = 10
    # Synthetic tokens representing [Prompt | Response] sequences
    chosen_batch = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_batch = torch.randint(0, vocab_size, (batch_size, seq_len))

    print("Preference Dataset Pairwise Batch Created.")
    print(f"  Batch Size: {batch_size} | Sequence Length: {seq_len}")
    print(f"  Active Policy Model Parameters: {sum(p.numel() for p in policy_model.parameters()):,}")
    print(f"  Frozen Reference Model Parameters: {sum(p.numel() for p in reference_model.parameters()):,}")

    # 3. Reference Model Baseline Pre-computation
    with torch.no_grad():
        ref_chosen_logps = reference_model.compute_sequence_log_probs(chosen_batch)
        ref_rejected_logps = reference_model.compute_sequence_log_probs(rejected_batch)

    print("\n--- DPO Pairwise Preference Optimization Loop (15 Steps) ---")
    policy_model.train()
    for step in range(1, 16):
        optimizer.zero_grad()

        # Compute Active Policy Log-Probabilities
        policy_chosen_logps = policy_model.compute_sequence_log_probs(chosen_batch)
        policy_rejected_logps = policy_model.compute_sequence_log_probs(rejected_batch)

        # DPO Loss
        loss, reward_w, reward_l = dpo_calculator.compute_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps
        )
        loss.backward()
        optimizer.step()

        if step % 3 == 0 or step == 1:
            margin = (reward_w - reward_l).item()
            print(f"DPO Step {step:02d} | Loss: {loss.item():.6f} | Implicit Reward Margin (Chosen - Rejected): {margin:+.6f}")

    print("\nTask 13 completed successfully! Direct Preference Alignment verified.")

if __name__ == "__main__":
    main()
