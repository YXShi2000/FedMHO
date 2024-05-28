# -*- coding: utf-8 -*-
import torch


def define_optimizer(conf, model, optimizer_name, use_lr_ml=None, arch="", lr=None):
    # define the param to optimize.
    params = [
        {
            "params": [value],
            "name": key,
            "weight_decay": conf.weight_decay if "bn" not in key else 0.0,
            "param_size": value.size(),
            "nelement": value.nelement(),
        }
        for key, value in model.named_parameters()
    ]

    # define the optimizer.
    model_lr = conf.lr if lr is None else lr
    if use_lr_ml:
        model_lr=conf.lr_ml
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=model_lr,
            momentum=conf.momentum_factor,
            nesterov=conf.use_nesterov,
            weight_decay=conf.weight_decay if "resnet" in arch else 0
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=conf.lr if lr is None else lr,
            betas=(conf.adam_beta_1, conf.adam_beta_2),
            eps=conf.adam_eps,
            weight_decay=conf.weight_decay,
        )
    else:
        raise NotImplementedError
    return optimizer
