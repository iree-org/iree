# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import unittest

import pytest
import torch
from iree.importers.torch import decompositions


class DecompTest(unittest.TestCase):
    def testDefault(self):
        table = decompositions.current_aot_decompositions()
        self.assertTrue(table)

    def testExtendToEmpty(self):
        with decompositions.extend_aot_decompositions(from_current=False) as t:
            self.assertFalse(t, msg=f"{t}")
            current_table = decompositions.current_aot_decompositions()
            self.assertFalse(current_table, msg=f"{current_table}")

    def testNestedExtend(self):
        initial_table = decompositions.current_aot_decompositions()
        with decompositions.extend_aot_decompositions(from_current=False) as _:
            with decompositions.extend_aot_decompositions(
                add_ops=[
                    torch.ops.aten.masked_fill.Tensor,
                    torch.ops.aten.masked_fill.Scalar,
                ]
            ):
                current_table = decompositions.current_aot_decompositions()
                self.assertEqual(2, len(current_table), msg=f"{current_table}")
                with decompositions.extend_aot_decompositions(
                    remove_ops=[
                        torch.ops.aten.masked_fill.Tensor,
                    ]
                ):
                    current_table = decompositions.current_aot_decompositions()
                    self.assertEqual(1, len(current_table), msg=f"{current_table}")
        current_table = decompositions.current_aot_decompositions()
        self.assertDictEqual(current_table, initial_table)


if __name__ == "__main__":
    args = sys.argv[1:]
    ret_code = pytest.main([__file__, "-vv"] + args)

    sys.exit(ret_code)
