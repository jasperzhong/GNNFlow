import torch

def is_sorted(tensor: torch.Tensor):
    return torch.all(torch.ge(tensor[1:], tensor[:-1]))
