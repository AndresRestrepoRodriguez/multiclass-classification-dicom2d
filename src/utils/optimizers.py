import torch.optim as optim
from typing import Any, Dict


def define_adam_optimizer(parameters: Any, **kargs: Dict[str, Any]) -> optim.Adam:
    """
    Defines and returns an Adam optimizer with the given parameters and additional arguments.

    Args:
        parameters (Any): Model parameters to optimize. Typically, this is obtained using `model.parameters()`.
        **kargs (Dict[str, Any]): Additional keyword arguments for configuring the Adam optimizer 
                                  (e.g., learning rate, weight decay).

    Returns:
        optim.Adam: The Adam optimizer initialized with the provided parameters and configurations.
    """
    optimizer = optim.Adam(parameters, **kargs)
    return optimizer