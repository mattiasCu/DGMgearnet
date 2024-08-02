import torch
import numpy as np

LARGE_NUMBER = 1.e10

def select_from_edge_candidates(scores: torch.Tensor, k: int):
    Batch, Nmax, ensemble = scores.shape
    if k >= ensemble:
        return scores.new_ones(scores.shape)

    # 在最后一个维度上取 topk
    topk_values, _ = torch.topk(scores, k, dim=-1, largest=True, sorted=True)

    # 选择每个位置的第 k 个最大值
    thresh = topk_values[..., -1]  # 形状 (Batch, Nmax)

    # 使 thresh 的形状与 scores 匹配
    thresh = thresh.unsqueeze(-1)  # 形状 (Batch, Nmax, 1)

    # 创建掩码
    mask = (scores >= thresh).to(torch.float)
    return mask
