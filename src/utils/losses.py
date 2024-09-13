from torch import nn


def define_BCE_logits_loss() -> nn.BCEWithLogitsLoss:
    """
    Defines and returns a binary cross-entropy loss function with logits.

    BCEWithLogitsLoss combines a sigmoid layer with the binary cross-entropy loss 
    in a numerically stable way. This is typically used for binary classification problems.

    Returns:
        nn.BCEWithLogitsLoss: A loss function for binary classification tasks with logits.
    """
    criterion = nn.BCEWithLogitsLoss()
    return criterion


def define_NLL_loss() -> nn.NLLLoss:
    """
    Defines and returns a negative log-likelihood (NLL) loss function.

    NLLLoss is typically used for classification tasks where the model outputs
    log-probabilities (log_softmax). It is often used in conjunction with log_softmax 
    as the output layer of the model.

    Returns:
        nn.NLLLoss: A loss function for classification tasks using log-probabilities.
    """
    criterion = nn.NLLLoss()
    return criterion


def define_CrossEntropy_loss() -> nn.CrossEntropyLoss:
    """
    Defines and returns a cross-entropy loss function.

    CrossEntropyLoss combines `log_softmax` and negative log-likelihood loss in one function.
    It is commonly used for multi-class classification tasks where the model outputs raw logits.

    Returns:
        nn.CrossEntropyLoss: A loss function for multi-class classification tasks.
    """
    criterion = nn.CrossEntropyLoss()
    return criterion