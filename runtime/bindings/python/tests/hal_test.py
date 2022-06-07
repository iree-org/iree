# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.runtime
import numpy as np
import unittest


class NonDeviceHalTest(unittest.TestCase):

  def testEnums(self):
    print("MemoryType:", iree.runtime.MemoryType)
    print("HOST_VISIBLE:", int(iree.runtime.MemoryType.HOST_VISIBLE))

    # Enum and/or operations on BufferCompatibility.
    self.assertEqual(
        iree.runtime.BufferCompatibility.IMPORTABLE |
        iree.runtime.BufferCompatibility.EXPORTABLE,
        int(iree.runtime.BufferCompatibility.IMPORTABLE) |
        int(iree.runtime.BufferCompatibility.EXPORTABLE))
    self.assertEqual(
        iree.runtime.BufferCompatibility.EXPORTABLE &
        iree.runtime.BufferCompatibility.EXPORTABLE,
        int(iree.runtime.BufferCompatibility.EXPORTABLE))

    # Enum and/or operations on BufferUsage.
    self.assertEqual(
        iree.runtime.BufferUsage.CONSTANT | iree.runtime.BufferUsage.TRANSFER,
        int(iree.runtime.BufferUsage.CONSTANT) |
        int(iree.runtime.BufferUsage.TRANSFER))
    self.assertEqual(
        iree.runtime.BufferUsage.CONSTANT & iree.runtime.BufferUsage.CONSTANT,
        int(iree.runtime.BufferUsage.CONSTANT))

    # Enum and/or operations on MemoryAccess.
    self.assertEqual(
        iree.runtime.MemoryAccess.READ | iree.runtime.MemoryAccess.WRITE,
        int(iree.runtime.MemoryAccess.READ) |
        int(iree.runtime.MemoryAccess.WRITE))
    self.assertEqual(
        iree.runtime.MemoryAccess.ALL & iree.runtime.MemoryAccess.READ,
        int(iree.runtime.MemoryAccess.READ))

    # Enum and/or operations on MemoryType.
    self.assertEqual(
        iree.runtime.MemoryType.TRANSIENT |
        iree.runtime.MemoryType.HOST_VISIBLE,
        int(iree.runtime.MemoryType.TRANSIENT) |
        int(iree.runtime.MemoryType.HOST_VISIBLE))
    self.assertEqual(
        iree.runtime.MemoryType.TRANSIENT & iree.runtime.MemoryType.TRANSIENT,
        int(iree.runtime.MemoryType.TRANSIENT))


class DeviceHalTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.driver = iree.runtime.HalDriver.create("local-task")
    self.device = self.driver.create_default_device()
    self.allocator = self.device.allocator

  def testTrim(self):
    self.allocator.trim()
    # Just running is sufficient.

  def testStatistics(self):
    stats_dict = self.allocator.statistics
    stats_str = self.allocator.formatted_statistics
    if self.allocator.has_statistics:
      self.assertIn("host_bytes_peak", stats_dict)
      self.assertIn("host_bytes_allocated", stats_dict)
      self.assertIn("host_bytes_freed", stats_dict)
      self.assertIn("device_bytes_peak", stats_dict)
      self.assertIn("device_bytes_allocated", stats_dict)
      self.assertIn("device_bytes_freed", stats_dict)
      self.assertIn("HOST_LOCAL", stats_str)

  def testQueryCompatibility(self):
    compat = self.allocator.query_compatibility(
        memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
        allowed_usage=iree.runtime.BufferUsage.CONSTANT,
        intended_usage=iree.runtime.BufferUsage.CONSTANT |
        iree.runtime.BufferUsage.TRANSFER,
        allocation_size=1024)
    print("COMPAT:", compat)
    self.assertTrue(
        bool(compat & int(iree.runtime.BufferCompatibility.ALLOCATABLE)),
        "should be allocatable")
    self.assertTrue(
        bool(compat & int(iree.runtime.BufferCompatibility.IMPORTABLE)),
        "should be importable")
    self.assertTrue(
        bool(compat & int(iree.runtime.BufferCompatibility.EXPORTABLE)),
        "should be exportable")

  def testAllocateBuffer(self):
    buffer = self.allocator.allocate_buffer(
        memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
        allowed_usage=iree.runtime.BufferUsage.CONSTANT,
        allocation_size=13)
    print("BUFFER:", buffer)

  def testAllocateBufferCopy(self):
    ary = np.zeros([3, 4], dtype=np.int32) + 2
    buffer = self.allocator.allocate_buffer_copy(
        memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
        allowed_usage=iree.runtime.BufferUsage.CONSTANT,
        buffer=ary)
    self.assertEqual(
        repr(buffer),
        "<HalBuffer 48 bytes (at offset 0 into 48), memory_type=DEVICE_LOCAL|HOST_VISIBLE, allowed_access=ALL, allowed_usage=CONSTANT|TRANSFER|MAPPING>"
    )

  def testAllocateBufferViewCopy(self):
    ary = np.zeros([3, 4], dtype=np.int32) + 2
    buffer = self.allocator.allocate_buffer_copy(
        memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
        allowed_usage=iree.runtime.BufferUsage.CONSTANT,
        buffer=ary,
        element_type=iree.runtime.HalElementType.SINT_32)
    self.assertEqual(
        repr(buffer),
        "<HalBufferView (3, 4), element_type=0x20000011, 48 bytes (at offset 0 into 48), memory_type=DEVICE_LOCAL|HOST_VISIBLE, allowed_access=ALL, allowed_usage=CONSTANT|TRANSFER|MAPPING>"
    )


if __name__ == "__main__":
  unittest.main()
