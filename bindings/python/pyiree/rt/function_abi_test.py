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

from absl import logging
from absl.testing import absltest

import numpy as np
from pyiree import rt

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


class HostTypeFactory(absltest.TestCase):

  def test_baseclass(self):
    htf = rt.HostTypeFactory()
    logging.info("HostTypeFactory: %s", htf)


class FunctionAbiTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    driver_names = rt.HalDriver.query()
    for driver_name in driver_names:
      logging.info("Try to create driver: %s", driver_name)
      try:
        cls.driver = rt.HalDriver.create(driver_name)
        cls.device = cls.driver.create_default_device()
      except Exception:
        logging.error("Could not create driver: %s", driver_name)
      else:
        break

  def setUp(self):
    super().setUp()
    self.htf = rt.HostTypeFactory.get_numpy()

  def test_static_arg_success(self):
    fabi = rt.FunctionAbi(self.device, self.htf,
                          ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    logging.info("fabi: %s", fabi)
    self.assertEqual(
        "<FunctionAbi (Buffer<float32[10x128x64]>) -> "
        "(Buffer<sint32[32x8x64]>)>", repr(fabi))
    self.assertEqual(1, fabi.raw_input_arity)
    self.assertEqual(1, fabi.raw_result_arity)

    arg = np.zeros((10, 128, 64), dtype=np.float32)
    packed = fabi.raw_pack_inputs([arg])
    logging.info("packed: %s", packed)
    self.assertEqual("<VmVariantList(1): [HalBufferView(10x128x64:0x3000020)]>",
                     repr(packed))

  def test_static_result_success(self):
    fabi = rt.FunctionAbi(self.device, self.htf,
                          ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    arg = np.zeros((10, 128, 64), dtype=np.float32)
    f_args = fabi.raw_pack_inputs([arg])
    f_results = fabi.allocate_results(f_args)
    logging.info("f_results: %s", f_results)
    self.assertEqual("<VmVariantList(1): [HalBufferView(32x8x64:0x1000020)]>",
                     repr(f_results))
    py_result, = fabi.raw_unpack_results(f_results)
    self.assertEqual(np.int32, py_result.dtype)
    self.assertEqual((32, 8, 64), py_result.shape)

  def test_dynamic_alloc_result_success(self):
    fabi = rt.FunctionAbi(self.device, self.htf,
                          ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    arg = np.zeros((10, 128, 64), dtype=np.float32)
    f_args = fabi.raw_pack_inputs([arg])
    f_results = fabi.allocate_results(f_args, static_alloc=False)
    logging.info("f_results: %s", f_results)
    self.assertEqual("<VmVariantList(0): []>", repr(f_results))

  def test_dynamic_arg_success(self):
    fabi = rt.FunctionAbi(self.device, self.htf,
                          ATTRS_1ARG_FLOAT32_DYNX128X64_TO_SINT32_DYNX8X64_V1)
    logging.info("fabi: %s", fabi)
    self.assertEqual(
        "<FunctionAbi (Buffer<float32[?x128x64]>) -> "
        "(Buffer<sint32[?x8x64]>)>", repr(fabi))
    self.assertEqual(1, fabi.raw_input_arity)
    self.assertEqual(1, fabi.raw_result_arity)

    arg = np.zeros((10, 128, 64), dtype=np.float32)
    packed = fabi.raw_pack_inputs([arg])
    logging.info("packed: %s", packed)
    self.assertEqual("<VmVariantList(1): [HalBufferView(10x128x64:0x3000020)]>",
                     repr(packed))

  def test_static_arg_rank_mismatch(self):
    fabi = rt.FunctionAbi(self.device, self.htf,
                          ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    logging.info("fabi: %s", fabi)
    arg = np.zeros((10,), dtype=np.float32)
    with self.assertRaisesRegex(
        ValueError,
        re.escape("Mismatched buffer rank (received: 1, expected: 3)")):
      fabi.raw_pack_inputs([arg])

  def test_static_arg_eltsize_mismatch(self):
    fabi = rt.FunctionAbi(self.device, self.htf,
                          ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    logging.info("fabi: %s", fabi)
    arg = np.zeros((10, 128, 64), dtype=np.float64)
    with self.assertRaisesRegex(
        ValueError,
        re.escape("Mismatched buffer item size (received: 8, expected: 4)")):
      fabi.raw_pack_inputs([arg])

  def test_static_arg_dtype_mismatch(self):
    fabi = rt.FunctionAbi(self.device, self.htf,
                          ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    logging.info("fabi: %s", fabi)
    arg = np.zeros((10, 128, 64), dtype=np.int32)
    with self.assertRaisesRegex(
        ValueError,
        re.escape("Mismatched buffer format (received: i, expected: f)")):
      fabi.raw_pack_inputs([arg])

  def test_static_arg_static_dim_mismatch(self):
    fabi = rt.FunctionAbi(self.device, self.htf,
                          ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    logging.info("fabi: %s", fabi)
    arg = np.zeros((10, 32, 64), dtype=np.float32)
    with self.assertRaisesRegex(
        ValueError,
        re.escape("Mismatched buffer dim (received: 32, expected: 128)")):
      fabi.raw_pack_inputs([arg])


if __name__ == "__main__":
  absltest.main()
