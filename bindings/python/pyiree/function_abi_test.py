# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the function abi."""

import re

from absl.testing import absltest

import numpy as np
import pyiree

ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1 = (
    ("fv", "1"),
    # Equiv to:
    # (Buffer<float32[10x128x64]>) -> (Buffer<sint32[32x8x64]>)
    ("f", "I15!B11!d10d128d64R15!B11!t6d32d8d64"),
)

ATTRS_1ARG_FLOAT32_DYNX128X64_TO_SINT32_DYNX8X64_V1 = (
    ("fv", "1"),
    # Equiv to:
    # (Buffer<float32[?x128x64]>) -> (Buffer<sint32[?x8x64]>)
    ("f", "I15!B11!d-1d128d64R15!B11!t6d-1d8d64"),
)


class FunctionAbiTest(absltest.TestCase):

  def test_static_arg_success(self):
    rt_policy = pyiree.binding.rt.Policy()
    rt_instance = pyiree.binding.rt.Instance()
    rt_context = pyiree.binding.rt.Context(rt_instance, rt_policy)
    fabi = pyiree.binding.function_abi.create(
        ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    print(fabi)
    self.assertEqual(
        "<FunctionAbi (Buffer<float32[10x128x64]>) -> "
        "(Buffer<sint32[32x8x64]>)>", repr(fabi))
    self.assertEqual(1, fabi.raw_input_arity)
    self.assertEqual(1, fabi.raw_result_arity)

    arg = np.zeros((10, 128, 64), dtype=np.float32)
    packed = fabi.raw_pack_inputs(rt_context, [arg])
    print(packed)
    self.assertEqual("<FunctionArgVariantList(1): [HalBuffer(327680)]>",
                     repr(packed))

  def test_dynamic_arg_success(self):
    rt_policy = pyiree.binding.rt.Policy()
    rt_instance = pyiree.binding.rt.Instance()
    rt_context = pyiree.binding.rt.Context(rt_instance, rt_policy)
    fabi = pyiree.binding.function_abi.create(
        ATTRS_1ARG_FLOAT32_DYNX128X64_TO_SINT32_DYNX8X64_V1)
    print(fabi)
    self.assertEqual(
        "<FunctionAbi (Buffer<float32[?x128x64]>) -> "
        "(Buffer<sint32[?x8x64]>)>", repr(fabi))
    self.assertEqual(1, fabi.raw_input_arity)
    self.assertEqual(1, fabi.raw_result_arity)

    arg = np.zeros((10, 128, 64), dtype=np.float32)
    packed = fabi.raw_pack_inputs(rt_context, [arg])
    print(packed)
    self.assertEqual(
        "<FunctionArgVariantList(1): [HalBuffer(327680, dynamic_dims=[10])]>",
        repr(packed))

  def test_static_arg_rank_mismatch(self):
    rt_policy = pyiree.binding.rt.Policy()
    rt_instance = pyiree.binding.rt.Instance()
    rt_context = pyiree.binding.rt.Context(rt_instance, rt_policy)
    fabi = pyiree.binding.function_abi.create(
        ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    print(fabi)
    arg = np.zeros((10,), dtype=np.float32)
    with self.assertRaisesRegex(
        ValueError,
        re.escape("Mismatched buffer rank (received: 1, expected: 3)")):
      fabi.raw_pack_inputs(rt_context, [arg])

  def test_static_arg_eltsize_mismatch(self):
    rt_policy = pyiree.binding.rt.Policy()
    rt_instance = pyiree.binding.rt.Instance()
    rt_context = pyiree.binding.rt.Context(rt_instance, rt_policy)
    fabi = pyiree.binding.function_abi.create(
        ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    print(fabi)
    arg = np.zeros((10, 128, 64), dtype=np.float64)
    with self.assertRaisesRegex(
        ValueError,
        re.escape("Mismatched buffer item size (received: 8, expected: 4)")):
      fabi.raw_pack_inputs(rt_context, [arg])

  def test_static_arg_dtype_mismatch(self):
    rt_policy = pyiree.binding.rt.Policy()
    rt_instance = pyiree.binding.rt.Instance()
    rt_context = pyiree.binding.rt.Context(rt_instance, rt_policy)
    fabi = pyiree.binding.function_abi.create(
        ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    print(fabi)
    arg = np.zeros((10, 128, 64), dtype=np.int32)
    with self.assertRaisesRegex(
        ValueError,
        re.escape("Mismatched buffer format (received: i, expected: f)")):
      fabi.raw_pack_inputs(rt_context, [arg])

  def test_static_arg_static_dim_mismatch(self):
    rt_policy = pyiree.binding.rt.Policy()
    rt_instance = pyiree.binding.rt.Instance()
    rt_context = pyiree.binding.rt.Context(rt_instance, rt_policy)
    fabi = pyiree.binding.function_abi.create(
        ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    print(fabi)
    arg = np.zeros((10, 32, 64), dtype=np.float32)
    with self.assertRaisesRegex(
        ValueError,
        re.escape("Mismatched buffer dim (received: 32, expected: 128)")):
      fabi.raw_pack_inputs(rt_context, [arg])


if __name__ == "__main__":
  absltest.main()
