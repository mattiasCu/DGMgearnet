import torch
import sys
import os


from simple.simple_scheme import EdgeSIMPLEBatched, LARGE_NUMBER

k = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = EdgeSIMPLEBatched(k, device, logits_activation='None')

scores = torch.randn( 7, 55, 55).to(device)
scores[0, :15, :10] = -LARGE_NUMBER
scores[0, : 10, 30: ] = -LARGE_NUMBER
scores = scores*0.001
new_scores = model(scores)
new_mask, new_marginals = new_scores

print(new_mask.shape, new_marginals.shape)