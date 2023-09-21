# Lint as: python3
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/gguf.py
#
# To regenerate:
#  $ pip install gguf
#  $ cd runtime/src/iree/io/formats/gguf/testdata/
#  $ ./generate_gguf_files.py

import numpy as np
from gguf import GGUFWriter


def save_file(tensors, path):
    writer = GGUFWriter(path, "generic")

    writer.add_architecture()
    writer.add_custom_alignment(64)

    writer.add_uint32("metadata_uint32", 42)
    writer.add_string("metadata_str", "hello")
    writer.add_array("metadata_strs", ["a", "b", "c"])

    for key, value in tensors.items():
        writer.add_tensor(key, value)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()

    writer.close()


# no tensors
save_file({}, "empty.gguf")

# single tensor
save_file({"tensor0": np.ones((2, 2), dtype=np.float32)}, "single.gguf")

# multiple tensors
save_file(
    {
        "tensor0": np.ones((2, 2), dtype=np.float32),
        "tensor1": np.ones((1, 2), dtype=np.float32),
        "tensor2": np.ones((4, 3), dtype=np.float32),
    },
    "multiple.gguf",
)
