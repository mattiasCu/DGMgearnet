import torch
import numpy as np
from gumble_softmax.deterministic_scheme import select_from_edge_candidates


EPSILON = np.finfo(np.float32).tiny
LARGE_NUMBER = 1.e10

class GumbelSampler(torch.nn.Module):
    def __init__(self, k, tau=0.1, hard=True, policy=None):
        super(GumbelSampler, self).__init__()
        self.k = k
        self.hard = hard
        self.tau = tau
        self.adj = None   # for potential usage

    def forward(self, scores, train = True):
        repeat_sample = 1
        
        bsz, Nmax, ensemble = scores.shape
        flat_scores = scores.reshape(bsz * Nmax, ensemble)
        local_k = min(self.k, ensemble)

        # sample several times with
        flat_scores = flat_scores.repeat(repeat_sample, 1)

        m = torch.distributions.gumbel.Gumbel(flat_scores.new_zeros(flat_scores.shape),
                                              flat_scores.new_ones(flat_scores.shape))
        g = m.sample()
        flat_scores = flat_scores + g

        # continuous top k
        khot = flat_scores.new_zeros(flat_scores.shape)
        onehot_approx = flat_scores.new_zeros(flat_scores.shape)
        for i in range(local_k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON], device=flat_scores.device))
            flat_scores = flat_scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(flat_scores / self.tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = khot.new_zeros(khot.shape)
            val, ind = torch.topk(khot, local_k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        
        new_mask = res.reshape(repeat_sample, bsz, Nmax, ensemble).squeeze(0)

        return new_mask, khot

    @torch.no_grad()
    def validation(self, scores):

            
        mask = select_from_edge_candidates(scores, self.k)


        return mask[None].squeeze(0)

