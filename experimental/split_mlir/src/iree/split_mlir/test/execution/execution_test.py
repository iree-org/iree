# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from iree.split_mlir import extract_operation_list, execute_mlir_with_iree
from typing import List, Any
import os
import iree.runtime
import numpy as np


def assert_nested_array_equals(a: List[Any], b: List[Any]):
  assert a == b, f"{a} != {b}"


class ExecutionTest(unittest.TestCase):

  def test_extract_operation_list(self):
    expected_operation_list = [
        ("", []),
        ("call f1", [(0, 0), (0, 0)]),
        ("call f2", [(1, 0)]),
        ("return", [(0, 1), (2, 0)]),
    ]
    operation_list = extract_operation_list(mlir_file_path=os.path.join(
        os.path.dirname(__file__), "entry.mlir"),
                                            function_name="caller")
    assert_nested_array_equals(expected_operation_list, operation_list)

  def test_mlir_execution(self):
    mlir_path_function_pairs = [
        (os.path.join(os.path.dirname(__file__), "entry.mlir"), "caller"),
        (os.path.join(os.path.dirname(__file__), "f1.mlir"), "f1"),
        (os.path.join(os.path.dirname(__file__), "f2.mlir"), "main"),
    ]
    compile_kwargs = {
        "target_backends": ["llvm-cpu"],
    }
    device = iree.runtime.get_device("local-task")
    input = [np.array([1], dtype=np.float32), np.array([2], dtype=np.float32)]
    results = execute_mlir_with_iree(
        input=input,
        mlir_path_function_pairs=mlir_path_function_pairs,
        compile_kwargs=compile_kwargs,
        device=device)
    expected_output = [
        np.array([2], dtype=np.float32),
        np.array([4], dtype=np.float32)
    ]
    assert_nested_array_equals(results[-1], expected_output)


if __name__ == "__main__":
  unittest.main()
