# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compares the compiled erfinv decomposition against PyTorch.

torch-mlir cannot lower aten.erfinv, so the importer decomposes it into a
polynomial approximation. PyTorch's own erfinv is the reference.
"""

import math

import ml_dtypes
import numpy as np
import pytest
import torch
from iree.compiler import tools as iree_tools
from iree.importers.torch import import_programs
from iree.runtime import load_vm_flatbuffer


class Erfinv(torch.nn.Module):
    def forward(self, x):
        return torch.erfinv(x)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype == torch.bfloat16:
        # NumPy has no bfloat16 of its own, so torch cannot convert it.
        return tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return tensor.numpy()


def _to_torch(array: np.ndarray) -> torch.Tensor:
    if array.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(array.view(np.int16).copy()).view(torch.bfloat16)
    return torch.from_numpy(array.copy())


def _compile_and_run_erfinv(x: torch.Tensor) -> torch.Tensor:
    output = import_programs({"main": torch.export.export(Erfinv(), (x,))})
    output.lower_to_iree()
    vmfb = iree_tools.compile_str(
        str(output.mlir_module),
        extra_args=[
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
        ],
    )
    module = load_vm_flatbuffer(vmfb, driver="local-sync")
    return _to_torch(np.asarray(module.main(_to_numpy(x))))


def test_erfinv_matches_torch_over_the_open_interval():
    # The approximation switches from its central to its tail polynomial at
    # w = -log((1 - x)(1 + x)) == 5, that is at |x| == sqrt(1 - exp(-5)).
    boundary = torch.tensor(math.sqrt(1.0 - math.exp(-5.0)), dtype=torch.float32)
    below = torch.nextafter(boundary, torch.tensor(0.0))
    above = torch.nextafter(boundary, torch.tensor(1.0))
    # The polynomials overlap around the switch, so the boundary itself does
    # not show where it sits. |x| in [0.99, 0.999] spans w in about
    # [3.9, 6.2], past the overlap on both sides, where a moved switch is off
    # by about 1e-4.
    near_switch = torch.linspace(0.99, 0.999, 129)
    x = torch.cat(
        [
            torch.linspace(-0.99, 0.99, 1999),
            torch.tensor([0.0, 0.5, -0.5, 0.9, 0.95, -0.95]),
            near_switch,
            -near_switch,
            torch.stack([below, boundary, above, -below, -boundary, -above]),
            torch.tensor([0.999, -0.999]),
        ]
    )

    result = _compile_and_run_erfinv(x)

    torch.testing.assert_close(result, torch.erfinv(x), rtol=2e-6, atol=5e-7)


def test_erfinv_matches_torch_close_to_the_poles():
    x = torch.tensor([0.99999, -0.99999, 1 - 1e-7, -(1 - 1e-7)])

    result = _compile_and_run_erfinv(x)

    torch.testing.assert_close(result, torch.erfinv(x), rtol=1e-5, atol=5e-5)


def test_erfinv_matches_torch_at_the_poles_and_outside_the_interval():
    # erfinv is nan for |x| > 1 and a signed infinity at the poles. The tail
    # polynomial has a negative leading coefficient, so without the explicit
    # handling of the poles the result for +1 would be -inf.
    x = torch.tensor(
        [
            -float("inf"),
            -2.0,
            -1.0001,
            -1.0,
            1.0,
            1.0001,
            2.0,
            float("inf"),
            float("nan"),
        ],
        dtype=torch.float32,
    )

    result = _compile_and_run_erfinv(x)

    expected = torch.tensor(
        [
            float("nan"),
            float("nan"),
            float("nan"),
            -float("inf"),
            float("inf"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(result, expected, equal_nan=True)


def test_erfinv_keeps_the_sign_of_zero():
    x = torch.tensor([-0.0, 0.0], dtype=torch.float32)

    result = _compile_and_run_erfinv(x)

    assert torch.equal(result, torch.tensor([0.0, 0.0]))
    assert torch.signbit(result).tolist() == [True, False]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_erfinv_matches_torch_for_dtype(dtype):
    # Half precision is computed in f32 and cast back, since the polynomials
    # lose too much precision otherwise: computed in float16, some values of
    # this sweep are off by more than the tolerance. bfloat16 rounds |x| above
    # about 0.996 onto a pole, so the sweep stays below that.
    x = torch.linspace(-0.99, 0.99, 2001).to(dtype)

    result = _compile_and_run_erfinv(x)

    assert result.dtype == dtype
    torch.testing.assert_close(result, torch.erfinv(x))
