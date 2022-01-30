# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.runtime
import numpy as np
import unittest


class DeviceHalTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.driver = iree.runtime.HalDriver.create("vmvx")
    self.device = self.driver.create_default_device()
    self.allocator = self.device.allocator

  def testMetadataAttributes(self):
    init_ary = np.zeros([3, 4], dtype=np.int32) + 2
    buffer_view = self.allocator.allocate_buffer_copy(
        memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
        allowed_usage=iree.runtime.BufferUsage.CONSTANT,
        buffer=init_ary,
        element_type=iree.runtime.HalElementType.SINT_32)

    ary = iree.runtime.DeviceArray(self.device, buffer_view)
    self.assertEqual([3, 4], ary.shape)
    self.assertEqual(np.int32, ary.dtype)

  def testExplicitHostTransfer(self):
    init_ary = np.zeros([3, 4], dtype=np.int32) + 2
    buffer_view = self.allocator.allocate_buffer_copy(
        memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
        allowed_usage=iree.runtime.BufferUsage.CONSTANT,
        buffer=init_ary,
        element_type=iree.runtime.HalElementType.SINT_32)

    ary = iree.runtime.DeviceArray(self.device, buffer_view)
    self.assertEqual(repr(ary), "<IREE DeviceArray: shape=[3, 4], dtype=int32>")
    self.assertFalse(ary.is_host_accessible)

    # Explicit transfer.
    cp = ary.to_host()
    np.testing.assert_array_equal(cp, init_ary)
    self.assertTrue(ary.is_host_accessible)

  def testOverrideDtype(self):
    init_ary = np.zeros([3, 4], dtype=np.int32) + 2
    buffer_view = self.allocator.allocate_buffer_copy(
        memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
        allowed_usage=iree.runtime.BufferUsage.CONSTANT,
        buffer=init_ary,
        element_type=iree.runtime.HalElementType.SINT_32)

    ary = iree.runtime.DeviceArray(self.device,
                                   buffer_view,
                                   override_dtype=np.float32)

    # Explicit transfer.
    cp = ary.to_host()
    self.assertEqual(cp.dtype, np.float32)
    np.testing.assert_array_equal(cp, init_ary.astype(np.float32))
    self.assertTrue(ary.is_host_accessible)

  def testIllegalImplicitHostTransfer(self):
    init_ary = np.zeros([3, 4], dtype=np.int32) + 2
    buffer_view = self.allocator.allocate_buffer_copy(
        memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
        allowed_usage=iree.runtime.BufferUsage.CONSTANT,
        buffer=init_ary,
        element_type=iree.runtime.HalElementType.SINT_32)

    ary = iree.runtime.DeviceArray(self.device, buffer_view)
    # Implicit transfer.
    with self.assertRaises(ValueError):
      _ = np.asarray(ary)

  def testImplicitHostArithmetic(self):
    init_ary = np.zeros([3, 4], dtype=np.int32) + 2
    buffer_view = self.allocator.allocate_buffer_copy(
        memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
        allowed_usage=iree.runtime.BufferUsage.CONSTANT,
        buffer=init_ary,
        element_type=iree.runtime.HalElementType.SINT_32)

    ary = iree.runtime.DeviceArray(self.device,
                                   buffer_view,
                                   implicit_host_transfer=True)
    sum = ary + init_ary
    np.testing.assert_array_equal(sum, init_ary + 2)
    self.assertTrue(ary.is_host_accessible)

  def testArrayFunctions(self):
    init_ary = np.zeros([3, 4], dtype=np.float32) + 2
    buffer_view = self.allocator.allocate_buffer_copy(
        memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
        allowed_usage=iree.runtime.BufferUsage.CONSTANT,
        buffer=init_ary,
        element_type=iree.runtime.HalElementType.SINT_32)

    ary = iree.runtime.DeviceArray(self.device,
                                   buffer_view,
                                   implicit_host_transfer=True)
    f = np.isfinite(ary)
    self.assertTrue(f.all())


if __name__ == "__main__":
  unittest.main()
