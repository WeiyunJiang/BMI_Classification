import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)