import torch.optim as optim


def define_adam_optimizer(parameters, **kargs):
    optimizer = optim.Adam(parameters, **kargs)
    return optimizer