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
# pylint: disable=line-too-long
# pylint: disable=broad-except
"""Tests for the function abi."""

import re

from absl import logging
from absl.testing import absltest
import iree.runtime
import numpy as np

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

ATTRS_SIP_1LEVEL_DICT = (
    # Extracted from reflection attributes for mobilebert.
    # (via iree-dump-module).
    # Input dict of "input_ids", "input_mask", "segment_ids"
    # Output dict of "end_logits", "start_logits"
    # Raw signature is:
    #   (Buffer<sint32[1x384]>, Buffer<sint32[1x384]>, Buffer<sint32[1x384]>) -> (Buffer<float32[1x384]>, Buffer<float32[1x384]>)
    ("fv", "1"),
    ("f", "I34!B9!t6d1d384B9!t6d1d384B9!t6d1d384R19!B7!d1d384B7!d1d384"),
    ("abi", "sip"),
    ("sip",
     "I53!D49!K10!input_ids_1K11!input_mask_2K12!segment_ids_0R39!D35!K11!end_logits_0K13!start_logits_1"
    ),
)

ATTRS_SIP_LINEAR_2ARG = (
    # SIP form of a function that takes 2 args of Buffer<float32[1]> and
    # returns one of the same type/shape.
    ("fv", "1"),
    ("f", "I11!B3!d1B3!d1R6!B3!d1"),
    ("abi", "sip"),
    ("sip", "I12!S9!k0_0k1_1R3!_0"),
)


class HostTypeFactory(absltest.TestCase):

  def test_baseclass(self):
    htf = iree.runtime.HostTypeFactory()
    logging.info("HostTypeFactory: %s", htf)


class FunctionAbiTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    driver_names = iree.runtime.HalDriver.query()
    for driver_name in driver_names:
      logging.info("Try to create driver: %s", driver_name)
      try:
        cls.driver = iree.runtime.HalDriver.create(driver_name)
        cls.device = cls.driver.create_default_device()
      except Exception:
        logging.error("Could not create driver: %s", driver_name)
      else:
        break

  def setUp(self):
    super().setUp()
    self.htf = iree.runtime.HostTypeFactory.get_numpy()

  def test_sip_dict_arg_result_success(self):
    fabi = iree.runtime.FunctionAbi(self.device, self.htf,
                                    ATTRS_SIP_1LEVEL_DICT)
    self.assertEqual(
        "<FunctionAbi (Buffer<sint32[1x384]>, Buffer<sint32[1x384]>, Buffer<sint32[1x384]>) -> (Buffer<float32[1x384]>, Buffer<float32[1x384]>) SIP:'I53!D49!K10!input_ids_1K11!input_mask_2K12!segment_ids_0R39!D35!K11!end_logits_0K13!start_logits_1'>",
        repr(fabi))
    input_ids = np.zeros((1, 384), dtype=np.int32)
    input_mask = np.zeros((1, 384), dtype=np.int32)
    segment_ids = np.zeros((1, 384), dtype=np.int32)
    f_args = fabi.pack_inputs(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids)
    self.assertEqual(
        "<VmVariantList(3): [HalBufferView(1x384:0x1000020), HalBufferView(1x384:0x1000020), HalBufferView(1x384:0x1000020)]>",
        repr(f_args))
    f_results = fabi.allocate_results(f_args)
    logging.info("f_results: %s", f_results)
    self.assertEqual(
        "<VmVariantList(2): [HalBufferView(1x384:0x3000020), HalBufferView(1x384:0x3000020)]>",
        repr(f_results))
    py_result = fabi.unpack_results(f_results)
    start_logits = py_result["start_logits"]
    end_logits = py_result["end_logits"]
    self.assertEqual(np.float32, start_logits.dtype)
    self.assertEqual(np.float32, end_logits.dtype)
    self.assertEqual((1, 384), start_logits.shape)
    self.assertEqual((1, 384), end_logits.shape)

  def test_sip_linear_success(self):
    fabi = iree.runtime.FunctionAbi(self.device, self.htf,
                                    ATTRS_SIP_LINEAR_2ARG)
    self.assertEqual(
        "<FunctionAbi (Buffer<float32[1]>, Buffer<float32[1]>) -> (Buffer<float32[1]>) SIP:'I12!S9!k0_0k1_1R3!_0'>",
        repr(fabi))
    arg0 = np.zeros((1,), dtype=np.float32)
    arg1 = np.zeros((1,), dtype=np.float32)
    f_args = fabi.pack_inputs(arg0, arg1)
    self.assertEqual(
        "<VmVariantList(2): [HalBufferView(1:0x3000020), HalBufferView(1:0x3000020)]>",
        repr(f_args))
    f_results = fabi.allocate_results(f_args)
    logging.info("f_results: %s", f_results)
    self.assertEqual("<VmVariantList(1): [HalBufferView(1:0x3000020)]>",
                     repr(f_results))
    result = fabi.unpack_results(f_results)
    print("SINGLE RESULT:", result)
    self.assertEqual(np.float32, result.dtype)
    self.assertEqual((1,), result.shape)

  def test_static_arg_success(self):
    fabi = iree.runtime.FunctionAbi(
        self.device, self.htf,
        ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    logging.info("fabi: %s", fabi)
    self.assertEqual(
        "<FunctionAbi (Buffer<float32[10x128x64]>) -> "
        "(Buffer<sint32[32x8x64]>)>", repr(fabi))
    self.assertEqual(1, fabi.raw_input_arity)
    self.assertEqual(1, fabi.raw_result_arity)

    arg = np.zeros((10, 128, 64), dtype=np.float32)
    packed = fabi.pack_inputs(arg)
    logging.info("packed: %s", packed)
    self.assertEqual("<VmVariantList(1): [HalBufferView(10x128x64:0x3000020)]>",
                     repr(packed))

  def test_static_result_success(self):
    fabi = iree.runtime.FunctionAbi(
        self.device, self.htf,
        ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    arg = np.zeros((10, 128, 64), dtype=np.float32)
    f_args = fabi.pack_inputs(arg)
    f_results = fabi.allocate_results(f_args)
    logging.info("f_results: %s", f_results)
    self.assertEqual("<VmVariantList(1): [HalBufferView(32x8x64:0x1000020)]>",
                     repr(f_results))
    py_result = fabi.unpack_results(f_results)
    self.assertEqual(np.int32, py_result.dtype)
    self.assertEqual((32, 8, 64), py_result.shape)

  def test_dynamic_alloc_result_success(self):
    fabi = iree.runtime.FunctionAbi(
        self.device, self.htf,
        ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    arg = np.zeros((10, 128, 64), dtype=np.float32)
    f_args = fabi.pack_inputs(arg)
    f_results = fabi.allocate_results(f_args, static_alloc=False)
    logging.info("f_results: %s", f_results)
    self.assertEqual("<VmVariantList(0): []>", repr(f_results))

  def test_dynamic_arg_success(self):
    fabi = iree.runtime.FunctionAbi(
        self.device, self.htf,
        ATTRS_1ARG_FLOAT32_DYNX128X64_TO_SINT32_DYNX8X64_V1)
    logging.info("fabi: %s", fabi)
    self.assertEqual(
        "<FunctionAbi (Buffer<float32[?x128x64]>) -> "
        "(Buffer<sint32[?x8x64]>)>", repr(fabi))
    self.assertEqual(1, fabi.raw_input_arity)
    self.assertEqual(1, fabi.raw_result_arity)

    arg = np.zeros((10, 128, 64), dtype=np.float32)
    packed = fabi.pack_inputs(arg)
    logging.info("packed: %s", packed)
    self.assertEqual("<VmVariantList(1): [HalBufferView(10x128x64:0x3000020)]>",
                     repr(packed))

  def test_static_arg_rank_mismatch(self):
    fabi = iree.runtime.FunctionAbi(
        self.device, self.htf,
        ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    logging.info("fabi: %s", fabi)
    arg = np.zeros((10,), dtype=np.float32)
    with self.assertRaisesRegex(
        ValueError,
        re.escape("Mismatched buffer rank (received: 1, expected: 3)")):
      fabi.pack_inputs(arg)

  def test_static_arg_eltsize_mismatch(self):
    fabi = iree.runtime.FunctionAbi(
        self.device, self.htf,
        ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    logging.info("fabi: %s", fabi)
    arg = np.zeros((10, 128, 64), dtype=np.float64)
    with self.assertRaisesRegex(
        ValueError,
        re.escape("Mismatched buffer item size (received: 8, expected: 4)")):
      fabi.pack_inputs(arg)

  def test_static_arg_dtype_mismatch(self):
    fabi = iree.runtime.FunctionAbi(
        self.device, self.htf,
        ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    logging.info("fabi: %s", fabi)
    arg = np.zeros((10, 128, 64), dtype=np.int32)
    with self.assertRaisesRegex(
        ValueError,
        re.escape("Mismatched buffer format (received: i, expected: f)")):
      fabi.pack_inputs(arg)

  def test_static_arg_static_dim_mismatch(self):
    fabi = iree.runtime.FunctionAbi(
        self.device, self.htf,
        ATTRS_1ARG_FLOAT32_10X128X64_TO_SINT32_32X8X64_V1)
    logging.info("fabi: %s", fabi)
    arg = np.zeros((10, 32, 64), dtype=np.float32)
    with self.assertRaisesRegex(
        ValueError,
        re.escape("Mismatched buffer dim (received: 32, expected: 128)")):
      fabi.pack_inputs(arg)


if __name__ == "__main__":
  absltest.main()
