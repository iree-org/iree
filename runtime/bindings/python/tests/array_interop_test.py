# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import gc
import numpy as np
import unittest

import iree.runtime


class DeviceHalTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.device = iree.runtime.get_device("local-task")
    self.allocator = self.device.allocator
    # Make sure device setup maintains proper references.
    gc.collect()

  def testGcShutdownFiasco(self):
    init_ary = np.zeros([3, 4], dtype=np.int32) + 2
    ary = iree.runtime.asdevicearray(self.device, init_ary)

    # Drop all references to backing objects in reverse order to try to
    # trigger heap use-after-free on bad shutdown order.
    self.allocator = None
    gc.collect()
    self.device = None
    gc.collect()

    # Now drop the ary and make sure nothing crashes (which would indicate
    # a reference counting problem of some kind): The array should retain
    # everything that it needs to stay live.
    ary = None
    gc.collect()

  def testMetadataAttributes(self):
    init_ary = np.zeros([3, 4], dtype=np.int32) + 2
    ary = iree.runtime.asdevicearray(self.device, init_ary)
    self.assertEqual([3, 4], ary.shape)
    self.assertEqual(np.int32, ary.dtype)

  def testExplicitHostTransfer(self):
    init_ary = np.zeros([3, 4], dtype=np.int32) + 2
    ary = iree.runtime.asdevicearray(self.device, init_ary)
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
        allowed_usage=iree.runtime.BufferUsage.DEFAULT,
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
    ary = iree.runtime.asdevicearray(self.device, init_ary)
    # Implicit transfer.
    with self.assertRaises(ValueError):
      _ = np.asarray(ary)

  def testImplicitHostArithmetic(self):
    init_ary = np.zeros([3, 4], dtype=np.int32) + 2
    ary = iree.runtime.asdevicearray(self.device,
                                     init_ary,
                                     implicit_host_transfer=True)
    sum = ary + init_ary
    np.testing.assert_array_equal(sum, init_ary + 2)
    self.assertTrue(ary.is_host_accessible)

  def testArrayFunctions(self):
    init_ary = np.zeros([3, 4], dtype=np.float32) + 2
    ary = iree.runtime.asdevicearray(self.device,
                                     init_ary,
                                     implicit_host_transfer=True)
    f = np.isfinite(ary)
    self.assertTrue(f.all())

  def testIteration(self):
    init_ary = np.array([0, 1, 2, 3, 4, 5])
    ary = iree.runtime.asdevicearray(self.device,
                                     init_ary,
                                     implicit_host_transfer=True)

    for index, value in enumerate(ary):
      self.assertEqual(index, value)

  def testSubscriptable(self):
    init_ary = np.array([0, 1, 2, 3, 4, 5])
    ary = iree.runtime.asdevicearray(self.device,
                                     init_ary,
                                     implicit_host_transfer=True)

    for index in range(0, 6):
      value = ary[index]
      self.assertEqual(index, value)

  def testReshape(self):
    init_ary = np.zeros([3, 4], dtype=np.float32) + 2
    ary = iree.runtime.asdevicearray(self.device,
                                     init_ary,
                                     implicit_host_transfer=True)
    reshaped = ary.reshape((4, 3))
    self.assertEqual((4, 3), reshaped.shape)

    np_reshaped = np.reshape(ary, (2, 2, 3))
    self.assertEqual((2, 2, 3), np_reshaped.shape)

  def testDeepcopy(self):
    init_ary = np.zeros([3, 4], dtype=np.float32) + 2
    orig_ary = iree.runtime.asdevicearray(self.device,
                                          init_ary,
                                          implicit_host_transfer=True)
    copy_ary = copy.deepcopy(orig_ary)
    self.assertIsNot(orig_ary, copy_ary)
    np.testing.assert_array_equal(orig_ary, copy_ary)

  def testAsType(self):
    init_ary = np.zeros([3, 4], dtype=np.int32) + 2
    orig_ary = iree.runtime.asdevicearray(self.device,
                                          init_ary,
                                          implicit_host_transfer=True)
    # Same dtype, no copy.
    i32_nocopy = orig_ary.astype(np.int32, copy=False)
    self.assertIs(orig_ary, i32_nocopy)

    # Same dtype, copy.
    i32_nocopy = orig_ary.astype(np.int32)
    self.assertIsNot(orig_ary, i32_nocopy)
    np.testing.assert_array_equal(orig_ary, i32_nocopy)

    # Different dtype, copy.
    f32_copy = orig_ary.astype(np.float32)
    self.assertIsNot(orig_ary, f32_copy)
    self.assertEqual(f32_copy.dtype, np.float32)
    np.testing.assert_array_equal(orig_ary.astype(np.float32), f32_copy)

  def testBool(self):
    init_ary = np.zeros([3, 4], dtype=np.bool_)
    init_ary[1] = True  # Set some non-zero value.
    ary = iree.runtime.asdevicearray(self.device, init_ary)
    self.assertEqual(repr(ary), "<IREE DeviceArray: shape=[3, 4], dtype=bool>")
    np.testing.assert_array_equal(ary.to_host(), init_ary)


if __name__ == "__main__":
  unittest.main()
