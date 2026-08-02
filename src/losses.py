import torch
import torch.nn as nn
import torch.nn.functional as F


class SeesawLoss(nn.Module):
    """
    Memory-efficient Seesaw Loss for large class counts (10,000+ classes).
    Avoids building dynamic N x N matrices per batch.
    """
    def __init__(self, cls_num_list: list, p: float = 0.8, q: float = 2.0, eps: float = 1e-2):
        super(SeesawLoss, self).__init__()
        self.p = p
        self.q = q
        self.eps = eps

        cls_num_tensor = torch.tensor(cls_num_list, dtype=torch.float32)
        self.register_buffer('cls_num_list', cls_num_tensor)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Mitigation Factor (S_ij) based on instance frequency
        num_list = self.cls_num_list
        # Broadcast ratio: N_j / N_i
        target_counts = num_list[targets].unsqueeze(1)  # [B, 1]
        all_counts = num_list.unsqueeze(0)             # [1, C]
        matrix = all_counts / target_counts             # [B, C]

        mitigation_factor = torch.pow(matrix, self.p)
        mitigation_factor = torch.where(
            matrix > 1.0, 
            mitigation_factor, 
            torch.ones_like(mitigation_factor)
        )

        # 2. Compensation Factor (C_ij) based on false positive probabilities
        probs = F.softmax(logits, dim=1)                # [B, C]
        self_probs = probs.gather(1, targets.unsqueeze(1)) # [B, 1]

        compensation_factor = torch.pow(probs / (self_probs + self.eps), self.q)
        compensation_factor = torch.where(
            compensation_factor > 1.0, 
            compensation_factor, 
            torch.ones_like(compensation_factor)
        )

        # Combined weights: [B, C]
        seesaw_weights = mitigation_factor * compensation_factor

        # Do not adjust the target class logit itself
        one_hot = F.one_hot(targets, num_classes=logits.size(1)).bool()
        seesaw_weights[one_hot] = 1.0

        adjusted_logits = logits + torch.log(seesaw_weights + 1e-8)
        return F.cross_entropy(adjusted_logits, targets)