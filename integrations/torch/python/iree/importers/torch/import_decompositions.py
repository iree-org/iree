# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The decompositions that bring a torch program into the form IREE imports."""

import contextlib
from collections.abc import Callable, Iterator

import torch
from torch._decomp import get_decompositions
from torch._export.utils import (
    _collect_all_valid_cia_ops_for_aten_namespace,
    _get_decomp_for_cia,
)

from iree.importers.torch.decompositions import extend_aot_decompositions

__all__ = [
    "IMPORT_DECOMPOSITION_OPS",
    "import_decomposition_table",
    "import_decompositions",
    "register_import_decomposition",
]

DecompositionTable = dict[torch._ops.OperatorBase, Callable]

_aten = torch.ops.aten


# torch-mlir has no lowering for aten.erfinv: the op only exists in its
# generated op definitions, with no conversion or decomposition.
# Coefficients from Giles, "Approximating the inverse error function" (2010).
_ERFINV_CENTRAL_COEFFS = (
    2.81022636e-08,
    3.43273939e-07,
    -3.5233877e-06,
    -4.39150654e-06,
    0.00021858087,
    -0.00125372503,
    -0.00417768164,
    0.246640727,
    1.50140941,
)
_ERFINV_TAIL_COEFFS = (
    -0.000200214257,
    0.000100950558,
    0.00134934322,
    -0.00367342844,
    0.00573950773,
    -0.0076224613,
    0.00943887047,
    1.00167406,
    2.83297682,
)


def _horner(w: torch.Tensor, coeffs: tuple[float, ...]) -> torch.Tensor:
    p = coeffs[0] * w + coeffs[1]
    for coeff in coeffs[2:]:
        p = p * w + coeff
    return p


def _erfinv(x: torch.Tensor) -> torch.Tensor:
    """Branchless erfinv, for torch-mlir which cannot lower aten.erfinv.

    Both polynomial branches are evaluated and selected with a ``where`` so the
    graph carries no data-dependent control flow. Half precision is computed in
    f32 and cast back, as the polynomials underflow otherwise.

    The coefficients are Giles' single precision approximation: accurate enough
    for f32 computation, but not an f64-quality erfinv even on f64 input.
    """
    dtype = x.dtype
    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    xc = x.to(compute_dtype)
    # (1 - x)(1 + x) stays in (0, 1] for |x| < 1, so w is non-negative; |x| == 1
    # gives inf (taking the tail branch) and |x| > 1 gives nan, both matching aten.
    w = -torch.log((1.0 - xc) * (1.0 + xc))
    central = _horner(w - 2.5, _ERFINV_CENTRAL_COEFFS)
    tail = _horner(torch.sqrt(w) - 3.0, _ERFINV_TAIL_COEFFS)
    result = torch.where(w < 5.0, central, tail) * xc
    # At |x| == 1 both w and the tail polynomial diverge, and the polynomial's
    # negative leading coefficient flips the sign. Pin the poles to aten's +-inf.
    poles = xc * float("inf")
    return torch.where(xc.abs() == 1.0, poles, result).to(dtype)


IMPORT_DECOMPOSITION_OPS = [
    _aten._scaled_dot_product_flash_attention_for_cpu.default,
    _aten.scaled_dot_product_attention.default,
    _aten._scaled_dot_product_attention_math.default,
    _aten._safe_softmax,
    _aten._weight_norm_interface,
    _aten.upsample_bicubic2d,
    _aten.upsample_bilinear2d,
    _aten.upsample_nearest2d,
    _aten.split,
    _aten._weight_norm.default,
    _aten.concat.default,
    _aten.__and__.Tensor,
    # Alias of aten.mul.Tensor. torch-mlir has no op for it, so without this it
    # reaches the importer as a generic torch.operator and fails to legalize.
    _aten.multiply.Tensor,
    _aten.norm_except_dim.default,
    # aten.diff is not supported in torch-mlir.
    _aten.diff.default,
    # Since torch 2.6 linear layers are no longer decomposed automatically.
    # torch-mlir can decompose them itself, but to a batch_matmul with a batch
    # size of one instead of a matmul.
    _aten.linear.default,
    # Part of the linear decomposition, becomes aten.mm or aten.bmm.
    _aten.matmul.default,
    _aten.lstm,
    _aten.gru,
    # Convolutions are decomposed into the generic aten.convolution. The
    # conversions to linalg and TOSA lift them to conv1d, conv2d or conv3d
    # again, but are only implemented for the generic form.
    _aten.conv1d,
    _aten.conv1d.default,
    _aten.conv2d,
    _aten.conv2d.default,
    _aten.conv3d,
    _aten.conv3d.default,
    # Like the convolutions, hardtanh is decomposed, to a clamp.
    _aten.hardtanh,
    _aten.hardtanh.default,
]

_CUSTOM_DECOMPOSITIONS: DecompositionTable = {
    # TODO: torch-mlir should handle this.
    _aten.concat.default: torch.cat,
    _aten.erfinv.default: _erfinv,
}


def register_import_decomposition(
    op: torch._ops.OperatorBase, decomposition: Callable
) -> None:
    """Make every later import decompose ``op`` with ``decomposition``.

    Meant for modules that are loaded to prepare an import, such as the ones
    the console tool takes with ``--import-module``. Code that calls the
    importer can scope a decomposition with ``extend_aot_decompositions``
    instead.
    """
    _CUSTOM_DECOMPOSITIONS[op] = decomposition


def import_decomposition_table() -> DecompositionTable:
    """Return the decompositions to apply before importing a program."""
    table = get_decompositions(IMPORT_DECOMPOSITION_OPS)
    cia_ops = _collect_all_valid_cia_ops_for_aten_namespace()
    for op in IMPORT_DECOMPOSITION_OPS:
        if op in cia_ops:
            table[op] = _get_decomp_for_cia(op)
    table.update(_CUSTOM_DECOMPOSITIONS)
    return table


@contextlib.contextmanager
def import_decompositions() -> Iterator[DecompositionTable]:
    """Context manager that makes exports and imports use the import table."""
    with extend_aot_decompositions(add_decomps=import_decomposition_table()) as table:
        yield table
