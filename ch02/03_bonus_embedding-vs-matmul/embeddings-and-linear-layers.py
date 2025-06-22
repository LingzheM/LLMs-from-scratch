import torch

idx = torch.tensor([2, 3, 1])

num_idx = max(idx) + 1

out_dim = 5

torch.manual_seed(123)

embedding = torch.nn.Embedding(num_idx, out_dim)

print(embedding.weight)

print(embedding(torch.tensor([1])))

print(embedding(idx))

onehot = torch.nn.functional.one_hot(idx)

print(onehot)

linear = torch.nn.Linear(num_idx, out_dim, bias=False)

print(linear.weight)

linear.weight = torch.nn.Parameter(embedding.weight.T)

print(linear(onehot.float()))