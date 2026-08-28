import importlib
from omegaconf import OmegaConf, DictConfig, ListConfig

import torch
import torch.distributed as dist
from typing import Union

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    try:
        module_imp = importlib.import_module(module, package=None)
        if reload:
            importlib.reload(module_imp)
        return getattr(module_imp, cls)
    except (ModuleNotFoundError, AttributeError) as error:
        raise ImportError(
            f"Cannot resolve configured object '{string}'."
        ) from error

def instantiate_from_config(config, **kwargs):
    if "module" not in config:
        raise KeyError("Expected key `module`.")

    cls = get_obj_from_str(config["module"])

    # Support legacy top-level arguments and the newer nested params format.
    # Runtime arguments (for example optimizer tensors) take precedence.
    legacy_params = {
        key: value
        for key, value in config.items()
        if key not in {"module", "params"}
    }
    configured_params = dict(legacy_params)
    configured_params.update(config.get("params", {}))
    configured_params.update(kwargs)
    return cls(**configured_params)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def instantiate_non_trainable_model(config):
    model = instantiate_from_config(config)
    model = model.eval()
    model.train = disabled_train
    for param in model.parameters():
        param.requires_grad = False

    return model
