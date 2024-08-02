import torch
import numpy as np

LARGE_NUMBER = 1.e10

def select_from_edge_candidates(scores: torch.Tensor, k: int):
    Batch, Nmax, ensemble = scores.shape
    if k >= Nmax:
        return scores.new_ones(scores.shape)

    thresh = torch.topk(scores, k, dim=1, largest=True, sorted=True).values[:, -1, :][:, None, :]
    mask = (scores >= thresh).to(torch.float)
    return mask
