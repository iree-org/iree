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
#  $ cd runtime/src/iree/io/formats/safetensors/testdata/
#  $ ./generate_safetensors_files.py

import torch
from safetensors.torch import save_file

# no tensors
save_file({}, "empty.safetensors")

# single tensor
save_file({"tensor0": torch.zeros((2, 2))}, "single.safetensors")

# multiple tensors
save_file(
    {
        "tensor0": torch.zeros((2, 2)),
        "tensor1": torch.zeros((1, 2)),
        "tensor2": torch.zeros((4, 3)),
    },
    "multiple.safetensors",
)
