# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import sys
import unittest

import pytest
import torch
from iree.importers.torch.exporter import export


class SimpleCache(torch.nn.Module):
    def __init__(self, max_size, dtype=torch.float32):
        super().__init__()
        self.register_buffer("cache", torch.zeros(max_size, dtype=dtype))

    def forward(self, input_pos, values):
        # input_pos: [S], values: [S]
        assert input_pos.shape[0] == values.shape[0]

        # Writing the values to the buffer at the specified positions
        # torch.ops.aten.index_put(self.cache, [input_pos], values)
        self.cache[input_pos] = values

        return self.cache


class ReadWriteReadCache(torch.nn.Module):
    def __init__(self, max_size, dtype=torch.float32):
        super().__init__()
        self.register_buffer("cache", torch.zeros(max_size, dtype=dtype))

    def forward(self, input_pos, values):
        # input_pos: [S], values: [S]
        assert input_pos.shape[0] == values.shape[0]
        cache_value_0 = self.cache[2].clone()
        # Writing the values to the buffer at the specified positions
        self.cache[input_pos] = values
        cache_value_1 = self.cache[2].clone()
        return self.cache, cache_value_0, cache_value_1


class WriteOnlyCache(torch.nn.Module):
    """A buffer written without first being read in the exported graph."""

    def __init__(self, max_size, dtype=torch.float32):
        super().__init__()
        self.register_buffer("cache", torch.zeros(max_size, dtype=dtype))

    def forward(self, values):
        self.cache.copy_(values * 2)
        return self.cache.sum()


class BufferTest(unittest.TestCase):
    def testMutableBuffer(self):
        max_size = 1025
        simple_cache = SimpleCache(max_size)

        input_pos = torch.tensor([2, 5, 7])
        values = torch.tensor([1.0, 2.0, 3.0])
        simple_cache(input_pos, values)
        exported_program = export(simple_cache, args=(input_pos, values))
        module_str = str(exported_program.mlir_module)
        self.assertIn(
            "util.global private mutable @__auto.constant_1025_torch.float32",
            module_str,
        )

    def testReadWriteReadMutableBuffer(self):
        max_size = 10
        simple_cache = ReadWriteReadCache(max_size)

        input_pos = torch.tensor([2, 5, 7])
        values = torch.tensor([1.0, 2.0, 3.0])
        simple_cache(input_pos, values)
        exported_program = export(simple_cache, args=(input_pos, values))
        module_str = str(exported_program.mlir_module)
        self.assertIn(
            "util.global private mutable @__auto.constant_10_torch.float32",
            module_str,
        )

    def testWriteOnlyMutableBuffer(self):
        max_size = 4
        write_only_cache = WriteOnlyCache(max_size)

        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        write_only_cache(values)
        exported_program = export(write_only_cache, args=(values,))
        module_str = str(exported_program.mlir_module)
        self.assertIn(
            "util.global private mutable @__auto.constant_4_torch.float32",
            module_str,
        )
        self.assertIn(
            "util.global.store",
            module_str,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = sys.argv[1:]
    ret_code = pytest.main([__file__, "-vv"] + args)

    sys.exit(ret_code)
