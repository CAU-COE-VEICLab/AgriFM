# --------------------------------------------------------
# Agri420K: A Large-Scale Benchmark Dataset for Agricultural Image Recognition
# Licensed under The MIT License [see LICENSE for details]
# Written by Yucong Wang and Guorun Li
# --------------------------------------------------------
import torchvision.models as models
import torch
from collections import OrderedDict
import re

def densenet201(num_classes=1000):
    return models.densenet201(pretrained=False, num_classes=num_classes)



def densenet_load_state_dict(model, weights, progress: bool) -> None:
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = weights.get_state_dict(progress=progress, check_hash=True)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict
    # model.load_state_dict(state_dict)
