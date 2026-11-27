# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
import threading
from collections.abc import Callable, Sequence
from typing import Union

import torch
from torch._decomp import get_decompositions, remove_decompositions

__all__ = ["current_aot_decompositions", "extend_aot_decompositions"]

DecompositionTable = dict[torch._ops.OperatorBase, Callable]
DecompositionOpsList = Sequence[
    Union[torch._ops.OperatorBase, torch._ops.OpOverloadPacket]
]

# Manages "scopes" for decompositions used. Each unique scope is an attribute on
# the _decomp_local. If the attribute is missing, then the default
# decompositions are used. The scope "aot" is used for all AOT cases.
_decomp_local = threading.local()


def _get_decomp_stack(scope: str) -> list[DecompositionTable]:
    try:
        return getattr(_decomp_local, scope)
    except AttributeError:
        stack: list[DecompositionTable] = []
        setattr(_decomp_local, scope, stack)
        return stack


def _current(scope: str) -> DecompositionTable:
    """Gets the current decomposition table (which may be the default)."""
    stack = _get_decomp_stack(scope)
    if stack:
        return dict(stack[-1])
    else:
        return dict(DEFAULT_DECOMPOSITION_TABLE)


@contextlib.contextmanager
def _extend_context_manager(
    scope: str,
    *,
    from_current: bool = True,
    add_ops: DecompositionOpsList | None = None,
    remove_ops: DecompositionOpsList | None = None,
    add_decomps: DecompositionTable | None = None,
):
    table: DecompositionTable
    if from_current:
        table = dict(_current(scope))
    else:
        table = {}
    if add_ops:
        table.update(get_decompositions(add_ops))
    if remove_ops:
        remove_decompositions(table, remove_ops)  # type: ignore
    if add_decomps:
        table.update(add_decomps)
    stack = _get_decomp_stack(scope)
    stack.append(table)
    try:
        yield table
    finally:
        popped = stack.pop()
        assert (
            popped is table
        ), "contextmanager unbalanced: popped different that pushed"


def _get_default_decomposition_ops() -> DecompositionOpsList:
    aten = torch.ops.aten
    # default decompositions pulled from SHARK / torch._decomp
    return [
        aten.embedding_dense_backward,
        aten.native_layer_norm_backward,
        aten.slice_backward,
        aten.select_backward,
        aten.norm.ScalarOpt_dim,
        aten.native_group_norm,
        aten.upsample_bilinear2d.vec,
        aten.split.Tensor,
        aten.native_layer_norm,
        aten.masked_fill.Tensor,
        aten.masked_fill.Scalar,
        aten.t,
        aten.addmm,
        # decompositions that aid us in handling nn.BatchNorm2d
        aten._native_batch_norm_legit_functional,
        aten._native_batch_norm_legit_no_training,
        aten._native_batch_norm_legit,
        aten._native_batch_norm_legit.no_stats,
        aten.squeeze.dims,
        # decompositions for miscellaneous ops that are not handled in torch-mlir but have available decompositions
        aten.soft_margin_loss,
        aten.im2col,
        aten._euclidean_dist,
        aten.index_copy,
        aten.index_copy_,
        aten.grid_sampler_2d,
        aten.log_sigmoid_forward,
        aten.unsafe_split.Tensor,
        aten.binary_cross_entropy,
        aten.dot,
        aten._adaptive_avg_pool2d,
        aten._prelu_kernel,
        aten.full,
        aten._log_softmax,
        aten.nll_loss_forward,
        aten.nll_loss_backward,
        aten._to_copy,
        aten._log_softmax_backward_data,
        aten.lift_fresh_copy.default,
        aten._unsafe_index.Tensor,
        aten.unbind.int,
        aten.linspace.Tensor_Tensor,
    ]


# Some older APIs still use an op list instead of a table.
DEFAULT_DECOMPOSITIONS: DecompositionOpsList = _get_default_decomposition_ops()

# The table of default decompositions.
DEFAULT_DECOMPOSITION_TABLE: DecompositionTable = get_decompositions(
    DEFAULT_DECOMPOSITIONS
)


def current_aot_decompositions() -> DecompositionTable:
    """Gets the current decomposition table for AOT."""
    return _current("aot")


def extend_aot_decompositions(
    *,
    from_current: bool = True,
    add_ops: DecompositionOpsList | None = None,
    remove_ops: DecompositionOpsList | None = None,
    add_decomps: DecompositionTable | None = None,
):
    """Context manager which extends the list of decompositions used for AOT."""
    return _extend_context_manager(
        "aot",
        from_current=from_current,
        add_ops=add_ops,
        remove_ops=remove_ops,
        add_decomps=add_decomps,
    )
