# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.runtime

import gc
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
        iree.runtime.BufferUsage.TRANSFER | iree.runtime.BufferUsage.MAPPING,
        int(iree.runtime.BufferUsage.TRANSFER) |
        int(iree.runtime.BufferUsage.MAPPING))
    self.assertEqual(
        iree.runtime.BufferUsage.TRANSFER & iree.runtime.BufferUsage.TRANSFER,
        int(iree.runtime.BufferUsage.TRANSFER))

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
        iree.runtime.MemoryType.DEVICE_LOCAL |
        iree.runtime.MemoryType.HOST_VISIBLE,
        int(iree.runtime.MemoryType.DEVICE_LOCAL) |
        int(iree.runtime.MemoryType.HOST_VISIBLE))
    self.assertEqual(
        iree.runtime.MemoryType.OPTIMAL & iree.runtime.MemoryType.OPTIMAL,
        int(iree.runtime.MemoryType.OPTIMAL))


class DeviceHalTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.device = iree.runtime.get_device("local-task")
    self.allocator = self.device.allocator
    gc.collect()

  def testTrim(self):
    self.allocator.trim()
    # Just running is sufficient.

  def testProfilingDefaults(self):
    self.device.begin_profiling()
    self.device.end_profiling()
    # Just running is sufficient.

  def testProfilingOptions(self):
    self.device.begin_profiling(mode="queue", file_path="foo.rdc")
    self.device.end_profiling()
    # Just running is sufficient.

  def testProfilingInvalidOptions(self):
    with self.assertRaisesRegex(ValueError, "unrecognized profiling mode"):
      self.device.begin_profiling(mode="SOMETHING THAT DOESN'T EXIST")

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
    compat = self.allocator.query_buffer_compatibility(
        memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
        allowed_usage=iree.runtime.BufferUsage.DEFAULT,
        intended_usage=iree.runtime.BufferUsage.DEFAULT,
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
        allowed_usage=iree.runtime.BufferUsage.DEFAULT,
        allocation_size=13)
    print("BUFFER:", buffer)

  def testAllocateBufferCopy(self):
    ary = np.zeros([3, 4], dtype=np.int32) + 2
    buffer = self.allocator.allocate_buffer_copy(
        memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
        allowed_usage=iree.runtime.BufferUsage.DEFAULT,
        buffer=ary)
    self.assertEqual(
        repr(buffer),
        "<HalBuffer 48 bytes (at offset 0 into 48), memory_type=DEVICE_LOCAL|HOST_VISIBLE, allowed_access=ALL, allowed_usage=TRANSFER|DISPATCH_STORAGE|MAPPING>"
    )

  def testAllocateBufferViewCopy(self):
    ary = np.zeros([3, 4], dtype=np.int32) + 2
    buffer = self.allocator.allocate_buffer_copy(
        memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
        allowed_usage=iree.runtime.BufferUsage.DEFAULT,
        buffer=ary,
        element_type=iree.runtime.HalElementType.SINT_32)
    self.assertEqual(
        repr(buffer),
        "<HalBufferView (3, 4), element_type=0x20000011, 48 bytes (at offset 0 into 48), memory_type=DEVICE_LOCAL|HOST_VISIBLE, allowed_access=ALL, allowed_usage=TRANSFER|DISPATCH_STORAGE|MAPPING>"
    )

class IndexedDeviceHalTest(DeviceHalTest):

  def setUp(self):
    super().setUp()
    self.driver = iree.runtime.get_driver("local-task")
    self.device = self.driver.create_device(0)



if __name__ == "__main__":
  unittest.main()
