import torch
import numpy as np

q = (torch.rand(2,3,4,5).to(torch.float32) - 0.5) * 1
k = (torch.rand(2,3,4,5).to(torch.float32) - 0.5) * 1
v = (torch.rand(2,3,4,5).to(torch.float32) - 0.5) * 1
mask = (torch.rand(2,3,4,4).to(torch.float32) - 0.5) > 0

print(torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask))