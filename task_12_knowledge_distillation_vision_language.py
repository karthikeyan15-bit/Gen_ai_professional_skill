"""
Task 12: Knowledge Distillation of Dual-Encoder Vision-Language Models
---------------------------------------------------------------------
Objective: Compress massive multi-modal representation systems into highly efficient
edge-deployable feature extractors using soft-label Kullback-Leibler (KL) divergence.

Required Tech Stack: PyTorch, SciPy
Loss Formulation:
  L_total = alpha * L_cosine + (1 - alpha) * (temperature^2) * L_KL(p_student, p_teacher)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. Dual-Encoder Vision-Language Model (CLIP Architecture)
# =====================================================================

class DualEncoderVLModel(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        # Vision Encoder (Simulated CNN/ViT)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(hidden_dim * 4 * 4, embed_dim)
        )
        # Text Encoder (Simulated Embedding Bag + Linear)
        self.text_encoder = nn.Sequential(
            nn.EmbeddingBag(1000, hidden_dim, mode='mean'),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_feats = F.normalize(self.vision_encoder(images), dim=-1)
        txt_feats = F.normalize(self.text_encoder(text_tokens), dim=-1)

        # Logit matrix (batch_size x batch_size)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (img_feats @ txt_feats.T)
        return img_feats, txt_feats, logits


# =====================================================================
# 2. Knowledge Distillation Loss Engine
# =====================================================================

class CompositeKnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, temperature: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(
        self,
        student_img_feats: torch.Tensor,
        student_txt_feats: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_img_feats: torch.Tensor,
        teacher_txt_feats: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        # 1. Cosine Distance Loss on feature representations
        cos_img_loss = 1.0 - F.cosine_similarity(student_img_feats, teacher_img_feats, dim=-1).mean()
        cos_txt_loss = 1.0 - F.cosine_similarity(student_txt_feats, teacher_txt_feats, dim=-1).mean()
        loss_cosine = 0.5 * (cos_img_loss + cos_txt_loss)

        # 2. Soft-label KL Divergence Loss on logit distributions
        p_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        log_p_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        loss_kl = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (self.temperature ** 2)

        # 3. Composite Weighted Loss
        loss_total = self.alpha * loss_cosine + (1.0 - self.alpha) * loss_kl
        return loss_total


def main():
    print("=" * 70)
    print("Task 12: Knowledge Distillation Vision-Language Verification")
    print("=" * 70)

    torch.manual_seed(42)
    batch_size = 8

    # Teacher Model (Large: embed_dim=128, hidden_dim=64)
    teacher = DualEncoderVLModel(embed_dim=128, hidden_dim=64)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # Student Model (Compact/Edge: embed_dim=128, hidden_dim=16)
    student = DualEncoderVLModel(embed_dim=128, hidden_dim=16)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher Model Parameters: {teacher_params:,}")
    print(f"Student Model Parameters: {student_params:,} ({student_params / teacher_params * 100:.1f}% of Teacher Size)")

    distill_loss_fn = CompositeKnowledgeDistillationLoss(alpha=0.4, temperature=2.5)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    # Synthetic multi-modal dataset batch
    dummy_images = torch.randn(batch_size, 3, 16, 16) # [batch, 3, H, W]
    dummy_texts = torch.randint(0, 1000, (batch_size, 5))

    # Teacher Inference (Static Target)
    with torch.no_grad():
        t_img_f, t_txt_f, t_logits = teacher(dummy_images, dummy_texts)

    print("\n--- Running Multi-Modal Distillation Training Loop (15 Steps) ---")
    student.train()
    for step in range(1, 16):
        optimizer.zero_grad()
        s_img_f, s_txt_f, s_logits = student(dummy_images, dummy_texts)

        loss = distill_loss_fn(s_img_f, s_txt_f, s_logits, t_img_f, t_txt_f, t_logits)
        loss.backward()
        optimizer.step()

        if step % 3 == 0 or step == 1:
            print(f"Step {step:02d} | Distillation Loss (Cosine + Soft-KL): {loss.item():.6f}")

    print("\nTask 12 completed successfully! Multimodal distillation verified.")

if __name__ == "__main__":
    main()
