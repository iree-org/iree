# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os

import functorch
from functorch._src.compile_utils import strip_overloads
import iree_torch
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import torch
import torch_mlir


# Dataset
X, y, coef = make_regression(n_features=3, coef=True)
X = torch.from_numpy(X).to(dtype=torch.float32)
y = torch.from_numpy(y).to(dtype=torch.float32)
X, X_test, y, y_test = train_test_split(X, y)

# Weights
w = torch.zeros(X.shape[1:])
b = torch.tensor(0.)


def _get_argparse():
    parser = argparse.ArgumentParser(
        description="Train and run a regression model.")
    parser.add_argument("--iree-backend",
        default="llvm-cpu",
        help="See https://iree-org.github.io/iree/deployment-configurations/ "
             "for the full list of options.")
    return parser


def _suppress_warnings():
    import warnings
    warnings.simplefilter("ignore")
    import os


def forward(w, b, X):
    return torch.matmul(X, w) + b


def mse(y_pred, y):
    err = y_pred - y
    return torch.mean(torch.square(err))


def loss_fn(w, b, X, y):
    y_pred = forward(w, b, X)
    return mse(y_pred, y)


grad_fn = functorch.grad(loss_fn, argnums=(0, 1))


def update(w, b, grad_w, grad_b):
    learning_rate = 0.05
    new_w = w - grad_w * learning_rate
    new_b = b - grad_b * learning_rate
    return new_w, new_b


def train(w, b, X, y):
    grad_w, grad_b = grad_fn(w, b, X, y)
    loss = loss_fn(w, b, X, y)
    return update(w, b, grad_w, grad_b) + (loss,)


def main():
    global w, b, X_test, y_test
    _suppress_warnings()
    args = _get_argparse().parse_args()

    #
    # Training
    #
    train_args = (w, b, X_test[:1], y_test[:1])
    graph = functorch.make_fx(train)(*train_args)

    # TODO: Remove once https://github.com/llvm/torch-mlir/issues/1495
    # is resolved.
    strip_overloads(graph)

    mlir = torch_mlir.compile(
        graph,
        train_args,
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
        use_tracing=False)

    vmfb = iree_torch.compile_to_vmfb(mlir, "llvm-cpu")
    with open(os.path.join(os.path.dirname(__file__),
                           "native_training.vmfb"), "wb") as f:
        f.write(vmfb)


if __name__ == "__main__":
    main()