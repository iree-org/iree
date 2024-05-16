#!/usr/bin/env python
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
#
# To generate for a different version:
#   * Find when the GGUF_VERSION changed in
#     https://github.com/ggerganov/llama.cpp/tree/master/gguf-py/gguf
#   * Find a release on https://pypi.org/project/gguf/#history before/after the
#     version changed (e.g. 0.4.0 is days before the 2-->3 change)
#   * Install that version: `pip install gguf==0.4.0`
#   * Run with a custom suffix: `./generate_gguf_files.py --suffix=_v2`

import argparse
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GGUF testdata file generator.")
    parser.add_argument(
        "--suffix",
        help="Suffix to save files with, e.g. `_v2` if exporting for an older GGUF version",
    )
    args = parser.parse_args()

    # no tensors
    save_file({}, f"empty{args.suffix}.gguf")

    # single tensor
    save_file(
        {"tensor0": np.ones((2, 2), dtype=np.float32)}, f"single{args.suffix}.gguf"
    )

    # multiple tensors
    save_file(
        {
            "tensor0": np.ones((2, 2), dtype=np.float32),
            "tensor1": np.ones((1, 2), dtype=np.float32),
            "tensor2": np.ones((4, 3), dtype=np.float32),
        },
        f"multiple{args.suffix}.gguf",
    )
