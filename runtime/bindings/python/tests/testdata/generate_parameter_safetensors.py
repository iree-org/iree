#!/usr/bin/env python
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# https://huggingface.co/docs/safetensors/index
#
# To regenerate:
#  $ pip install safetensors
#  $ ./runtime/bindings/python/tests/testdata/generate_parameter_safetensors.py

from pathlib import Path
import numpy as np
from safetensors.numpy import save_file

# multiple tensors
save_file(
    {
        "weight": np.zeros([30, 20], dtype=np.float32) + 2.0,
        "bias": np.zeros([30], dtype=np.float32) + 1.0,
    },
    Path(__file__).resolve().parent / "parameter_weight_bias_1.safetensors",
)
