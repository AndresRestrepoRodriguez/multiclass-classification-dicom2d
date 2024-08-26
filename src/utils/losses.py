from torch import nn


def define_BCE_logits_loss():
    criterion = nn.BCEWithLogitsLoss()
    return criterion


def define_NLL_loss():
    criterion = nn.NLLLoss()
    return criterion


def define_CrossEntropy_loss():
    criterion = nn.CrossEntropyLoss()
    return criterion