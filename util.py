# ------------------------------- util.py ---------------------------------
import torch
import torch, torch.nn.functional as F 

def freeze_module(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


@torch.inference_mode()
def pearson_corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    """x,y: (N,)"""
    vx = x - x.mean()
    vy = y - y.mean()
    corr = (vx * vy).sum() / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-8)
    return corr.item()

def _average_rank(x: torch.Tensor) -> torch.Tensor:
    """Rank 1-D tensor with average method for ties (standard Spearman ranking)."""
    n = x.shape[0]
    sorted_idx = x.argsort(stable=True)          # stable so ties keep their original order
    ranks = torch.empty(n, dtype=torch.float32)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and x[sorted_idx[j]] == x[sorted_idx[j + 1]]:
            j += 1
        avg_rank = (i + j) / 2.0                  # 0-based average rank
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


@torch.inference_mode()
def spearman_corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Spearman correlation between two 1D tensors.

    Uses average ranking for ties, matching the standard definition.
    """
    rx = _average_rank(x)
    ry = _average_rank(y)
    return pearson_corrcoef(rx, ry)


def dpo_loss(chosen_scores: torch.Tensor,
             rejected_scores: torch.Tensor,
             beta: float = 0.1,
             margin: float = 0.1) -> torch.Tensor:
    """
    Direct-Preference-Optimisation loss (pairwise logistic form).

    Args
    ----
    chosen_scores   : (B,) model scores for the preferred examples
    rejected_scores : (B,) model scores for the less-preferred examples
    beta            : temperature; larger → steeper preference curve

    Returns
    -------
    scalar loss     : -mean log σ[β( s_c – s_r )]
    """
    # logits = beta * (chosen_scores - rejected_scores)
    logits = beta * ((chosen_scores - rejected_scores) - margin)
    # identical to –log sigmoid but numerically stable
    return F.softplus(-logits).mean()
# -------------------------------------------------------------------------
