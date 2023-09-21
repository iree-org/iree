# Lint as: python3
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# https://huggingface.co/docs/safetensors/index
#
# To regenerate:
#  $ pip install torch safetensors
#  $ cd tools/test/
#  $ ./parameters.py

import torch
from safetensors.torch import save_file

save_file(
    {
        "a0": torch.arange(0, 4),
        "a1": torch.arange(4, 8),
    },
    "parameters_a.safetensors",
)

save_file(
    {
        "b0": torch.arange(8, 16),
        "b1": torch.arange(16, 32),
    },
    "parameters_b.safetensors",
)
