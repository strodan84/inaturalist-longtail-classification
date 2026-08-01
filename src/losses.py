import torch
import torch.nn as nn
import torch.nn.functional as F


class SeesawLoss(nn.Module):
    """
    Seesaw Loss dynamically balances positive and negative gradients
    for long-tailed data distributions using two factors:
      1. Mitigation Factor: Reduces penalties on rare (tail) classes.
      2. Compensation Factor: Increases penalties on frequent false positives.
    """
    def __init__(self, cls_num_list: list, p: float = 0.8, q: float = 2.0, eps: float = 1e-2):
        super(SeesawLoss, self).__init__()
        self.p = p
        self.q = q
        self.eps = eps

        cls_num_tensor = torch.tensor(cls_num_list, dtype=torch.float32)
        self.register_buffer('cls_num_list', cls_num_tensor)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_list = self.cls_num_list
        matrix = num_list.unsqueeze(0) / num_list.unsqueeze(1)

        # Mitigation Factor (S_ij)
        mitigation_factor = torch.pow(matrix, self.p)
        mitigation_factor = torch.where(
            matrix > 1.0, 
            mitigation_factor, 
            torch.ones_like(mitigation_factor)
        )

        # Class probability computation
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        self_probs = (probs * one_hot).sum(dim=1, keepdim=True)

        # Compensation Factor (C_ij)
        compensation_factor = torch.pow(probs / (self_probs + self.eps), self.q)
        compensation_factor = torch.where(
            compensation_factor > 1.0, 
            compensation_factor, 
            torch.ones_like(compensation_factor)
        )

        # Combined weight matrix (S_ij * C_ij)
        seesaw_weights = mitigation_factor.unsqueeze(0) * compensation_factor.unsqueeze(1)

        # Mask out diagonal
        identity = torch.eye(logits.size(1), device=logits.device).unsqueeze(0)
        seesaw_weights = seesaw_weights * (1.0 - identity) + identity

        adjusted_logits = logits + torch.log(seesaw_weights + 1e-8)
        return F.cross_entropy(adjusted_logits, targets)