import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from exercise_solutions import SelfAttention_v1, SelfAttention_v2
import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89], # Your
        [0.55, 0.87, 0.66], # journey
        [0.57, 0.85, 0.64], # starts
        [0.22, 0.58, 0.33], # with
        [0.77, 0.25, 0.10], # one
        [0.05, 0.80, 0.55]  # step
    ]
)

d_in, d_out = 3, 2


torch.manual_seed(123)

sa_v1 = SelfAttention_v1(d_in, d_out)
sa_v2 = SelfAttention_v2(d_in, d_out)

sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)

print(sa_v1(inputs))