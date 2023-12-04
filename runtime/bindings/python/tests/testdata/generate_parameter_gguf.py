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
from gguf import GGUFWriter


def save_file(tensors, path):
    writer = GGUFWriter(str(path), "generic")

    writer.add_architecture()
    writer.add_custom_alignment(64)

    for key, value in tensors.items():
        writer.add_tensor(key, value)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()

    writer.close()


# multiple tensors
save_file(
    {
        "weight": np.zeros([30, 20], dtype=np.float32) + 2.0,
        "bias": np.zeros([30], dtype=np.float32) + 1.0,
    },
    Path(__file__).resolve().parent / "parameter_weight_bias_1.gguf",
)
