# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import numpy as np
import unittest

import iree.runtime as rt


class VmTypesTest(unittest.TestCase):
    @classmethod
    def setUp(self):
        # Ensures types are registered.
        self.instance = rt.VmInstance()

    def testRefProtocol(self):
        lst1 = rt.VmVariantList(0)
        ref = lst1.__iree_vm_ref__
        ref2 = lst1.ref
        print(ref)
        print(ref2)
        self.assertEqual(ref, ref2)
        self.assertNotEqual(ref, False)
        lst2 = rt.VmVariantList.__iree_vm_cast__(ref)
        print(lst2)
        lst3 = ref.deref(rt.VmVariantList)
        print(lst3)
        self.assertEqual(lst1, lst2)
        self.assertEqual(lst2, lst3)
        self.assertNotEqual(lst1, False)
        self.assertTrue(ref.isinstance(rt.VmVariantList))

    def test_variant_list(self):
        l = rt.VmVariantList(5)
        logging.info("variant_list: %s", l)
        self.assertEqual(l.size, 0)

    def test_variant_list_i64(self):
        l = rt.VmVariantList(5)
        # Push a value that exceeds 32-bit range.
        l.push_int(10 * 1000 * 1000 * 1000)
        self.assertEqual(str(l), "<VmVariantList(1): [10000000000]>")

    def test_variant_list_buffer_view(self):
        device = rt.get_device("local-sync")
        ET = rt.HalElementType
        for dt, et in (
            (np.int8, ET.SINT_8),  #
            (np.int16, ET.SINT_16),  #
            (np.int32, ET.SINT_32),  #
            (np.int64, ET.SINT_64),  #
            (np.uint8, ET.UINT_8),  #
            (np.uint16, ET.UINT_16),  #
            (np.uint32, ET.UINT_32),  #
            (np.uint64, ET.UINT_64),  #
            (np.float16, ET.FLOAT_16),  #
            (np.float32, ET.FLOAT_32),  #
            (np.float64, ET.FLOAT_64),  #
            (np.complex64, ET.COMPLEX_64),  #
            (np.complex128, ET.COMPLEX_128),
        ):
            lst = rt.VmVariantList(5)
            ary1 = np.asarray([1, 2, 3, 4], dtype=dt)
            bv1 = device.allocator.allocate_buffer_copy(
                memory_type=rt.MemoryType.DEVICE_LOCAL,
                allowed_usage=(rt.BufferUsage.DEFAULT | rt.BufferUsage.MAPPING),
                device=device,
                buffer=ary1,
                element_type=et,
            )
            lst.push_ref(bv1)
            ary2 = rt.DeviceArray(
                device,
                lst.get_as_object(0, rt.HalBufferView),
                implicit_host_transfer=True,
            )
            np.testing.assert_array_equal(ary1, ary2)
            with self.assertRaises(IndexError):
                lst.get_as_object(1, rt.HalBufferView)

    def test_variant_list_buffer(self):
        device = rt.get_device("local-sync")
        lst = rt.VmVariantList(5)
        buffer = device.allocator.allocate_buffer(
            memory_type=rt.MemoryType.DEVICE_LOCAL,
            allowed_usage=rt.BufferUsage.DEFAULT,
            allocation_size=1024,
        )
        lst.push_ref(buffer)
        _ = (lst.get_as_object(0, rt.HalBuffer),)

    def test_variant_list_zero_rank_tensor_to_str(self):
        device = rt.get_device("local-sync")
        lst = rt.VmVariantList(1)
        array = np.array(1234, dtype=np.int32)
        buffer_view = device.allocator.allocate_buffer_copy(
            memory_type=rt.MemoryType.DEVICE_LOCAL,
            allowed_usage=(rt.BufferUsage.DEFAULT | rt.BufferUsage.MAPPING),
            device=device,
            buffer=array,
            element_type=rt.HalElementType.SINT_32,
        )
        lst.push_ref(buffer_view)
        self.assertEqual(str(lst), "<VmVariantList(1): [HalBufferView(:0x20000011)]>")

    def test_variant_list_fence_to_str(self):
        lst = rt.VmVariantList(1)
        fence = rt.HalFence(2)
        lst.push_ref(fence)
        self.assertEqual(str(lst), "<VmVariantList(1): [fence(0)]>")

    def test_variant_list_list(self):
        lst1 = rt.VmVariantList(5)
        lst2 = rt.VmVariantList(5)
        lst1.push_list(lst2)
        self.assertEqual("<VmVariantList(1): [List[]]>", str(lst1))
        lstout = lst1.get_as_list(0)
        self.assertEqual("<VmVariantList(0): []>", str(lstout))
        with self.assertRaises(IndexError):
            lst1.get_as_list(1)

    def test_vm_buffer(self):
        b1 = rt.VmBuffer(10, alignment=0, mutable=True)
        print(b1)
        contents = memoryview(b1)
        contents[0:] = b"0123456789"
        self.assertEqual(bytes(b1), b"0123456789")

    def test_vm_buffer_ro(self):
        b1 = rt.VmBuffer(10, alignment=16, mutable=False)
        contents = memoryview(b1)
        with self.assertRaises(TypeError):
            contents[0:] = b"0123456789"


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
