import torch
from gumble_softmax.gumble_topk_sample import GumbelSampler

LARGE_NUMBER = 1.e10

k = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GumbelSampler(k, tau=0.1, hard=True)

scores = torch.randn( 2, 5, 10).to(device)
scores[0, :2, :2] = -LARGE_NUMBER
new_scores = model(scores)
val = model.validation(scores)
new_mask, khot = new_scores
print(new_mask.shape, khot.shape)