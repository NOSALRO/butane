import torch

def gaussian(x):
    return (-x.square()).exp()

def squashing(x):
    return (9 / 8. * torch.sin(x)) + (1 / 8. * torch.sin(3. * x))
