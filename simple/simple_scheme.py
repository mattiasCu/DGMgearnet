import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simple.simple import Layer
from simple.tensor_utils import self_defined_softmax
from simple.deterministic_scheme import select_from_edge_candidates

LARGE_NUMBER = 1.e10

def logsigmoid(x):
    return -F.softplus(-x) + 1.e-7


class EdgeSIMPLEBatched(nn.Module):
    def __init__(self,
                 k,
                 device,
                 val_ensemble=1,
                 train_ensemble=1,
                 logits_activation=None):
        super(EdgeSIMPLEBatched, self).__init__()
        self.k = k
        self.device = device
        self.layer_configs = dict()
        self.adj = None  # for potential usage
        assert val_ensemble > 0 and train_ensemble > 0
        self.val_ensemble = val_ensemble
        self.train_ensemble = train_ensemble
        self.logits_activation = logits_activation

    def forward(self, scores, train = True):
        times_sampled = self.train_ensemble if train else self.val_ensemble

        bsz, window, ensemble = scores.shape
        flat_scores = scores.reshape(bsz * window, ensemble)
        target_size = ensemble
        local_k = min(self.k, ensemble)

        N = 2 ** math.ceil(math.log2(target_size))
        if (N, local_k) in self.layer_configs:
            layer = self.layer_configs[(N, local_k)]
        else:
            layer = Layer(N, local_k, self.device)
            self.layer_configs[(N, local_k)] = layer

        # padding
        flat_scores = torch.cat(
            [flat_scores,
             torch.full((flat_scores.shape[0], N - flat_scores.shape[1]),
                        fill_value=-LARGE_NUMBER,
                        dtype=flat_scores.dtype,
                        device=flat_scores.device)],
            dim=1)

        #default logits activation is none
        if self.logits_activation == 'None' or self.logits_activation is None:
            pass
        elif self.logits_activation == 'logsoftmax':
            # todo: it is bad heuristic to detect the padding
            masks = (flat_scores.detach() > - LARGE_NUMBER / 2).float()
            flat_scores = torch.vmap(self_defined_softmax, in_dims=0, out_dims=0)(flat_scores, masks)
            flat_scores = torch.log(flat_scores + 1 / LARGE_NUMBER)
        elif self.logits_activation == 'logsigmoid':
            # todo: sigmoid is not good, it makes large scores too similar, i.e. close to 1.
            flat_scores = logsigmoid(flat_scores)
        else:
            raise NotImplementedError

        # we potentially need to sample multiple times
        marginals = layer.log_pr(flat_scores)
        marginals = marginals.exp().permute(1, 0)
        # (times_sampled) x (B x E) x (N x N)
        samples = layer.sample(flat_scores, local_k, times_sampled)
        samples = (samples - marginals[None]).detach() + marginals[None]

        # unpadding
        samples = samples[..., :target_size]
        marginals = marginals[:, :target_size]

        # not need to add original edges, because we will do in contruct.py
        

        # VE x (B x E) x window -> VE x B x window x E
        new_mask = samples.reshape(times_sampled, bsz, window, ensemble).squeeze(0)
        # (B x E) x window -> B x window x E
        new_marginals = marginals.reshape(bsz, window, ensemble)

        return new_mask, new_marginals

    @torch.no_grad()
    def validation(self, scores):
        """
        during the inference we need to margin-out the stochasticity
        thus we do top-k once or sample multiple times

        Args:
            scores: shape B x N x N x E

        Returns:
            mask: shape B x N x N x (E x VE)

        """
        if self.val_ensemble == 1:
            _, marginals = self.forward(scores, False)

            # do deterministic top-k
            mask = select_from_edge_candidates(scores, self.k)

            return mask[None], marginals
        else:
            return self.forward(scores, False)