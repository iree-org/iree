# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.runtime

import gc
import numpy as np
import threading
import time
import unittest


class NonDeviceHalTest(unittest.TestCase):
    def testMemoryEnums(self):
        print("MemoryType:", iree.runtime.MemoryType)
        print("HOST_VISIBLE:", int(iree.runtime.MemoryType.HOST_VISIBLE))

        # Enum and/or operations on BufferCompatibility.
        self.assertEqual(
            iree.runtime.BufferCompatibility.IMPORTABLE
            | iree.runtime.BufferCompatibility.EXPORTABLE,
            int(iree.runtime.BufferCompatibility.IMPORTABLE)
            | int(iree.runtime.BufferCompatibility.EXPORTABLE),
        )
        self.assertEqual(
            iree.runtime.BufferCompatibility.EXPORTABLE
            & iree.runtime.BufferCompatibility.EXPORTABLE,
            int(iree.runtime.BufferCompatibility.EXPORTABLE),
        )

        # Enum and/or operations on BufferUsage.
        self.assertEqual(
            iree.runtime.BufferUsage.TRANSFER | iree.runtime.BufferUsage.MAPPING,
            int(iree.runtime.BufferUsage.TRANSFER)
            | int(iree.runtime.BufferUsage.MAPPING),
        )
        self.assertEqual(
            iree.runtime.BufferUsage.TRANSFER & iree.runtime.BufferUsage.TRANSFER,
            int(iree.runtime.BufferUsage.TRANSFER),
        )

        # Enum and/or operations on MemoryAccess.
        self.assertEqual(
            iree.runtime.MemoryAccess.READ | iree.runtime.MemoryAccess.WRITE,
            int(iree.runtime.MemoryAccess.READ) | int(iree.runtime.MemoryAccess.WRITE),
        )
        self.assertEqual(
            iree.runtime.MemoryAccess.ALL & iree.runtime.MemoryAccess.READ,
            int(iree.runtime.MemoryAccess.READ),
        )

        # Enum and/or operations on MemoryType.
        self.assertEqual(
            iree.runtime.MemoryType.DEVICE_LOCAL | iree.runtime.MemoryType.HOST_VISIBLE,
            int(iree.runtime.MemoryType.DEVICE_LOCAL)
            | int(iree.runtime.MemoryType.HOST_VISIBLE),
        )
        self.assertEqual(
            iree.runtime.MemoryType.OPTIMAL & iree.runtime.MemoryType.OPTIMAL,
            int(iree.runtime.MemoryType.OPTIMAL),
        )

    def testElementTypeEnums(self):
        i8 = iree.runtime.HalElementType.INT_8
        i4 = iree.runtime.HalElementType.INT_4
        self.assertTrue(iree.runtime.HalElementType.is_byte_aligned(i8))
        self.assertFalse(iree.runtime.HalElementType.is_byte_aligned(i4))
        self.assertEqual(1, iree.runtime.HalElementType.dense_byte_count(i8))


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
        self.device.flush_profiling()
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
            allocation_size=1024,
        )
        print("COMPAT:", compat)
        self.assertTrue(
            bool(compat & int(iree.runtime.BufferCompatibility.ALLOCATABLE)),
            "should be allocatable",
        )
        self.assertTrue(
            bool(compat & int(iree.runtime.BufferCompatibility.IMPORTABLE)),
            "should be importable",
        )
        self.assertTrue(
            bool(compat & int(iree.runtime.BufferCompatibility.EXPORTABLE)),
            "should be exportable",
        )

    def testAllocateBuffer(self):
        buffer = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=13,
        )
        print("BUFFER:", buffer)

    def testBufferViewConstructor(self):
        buffer = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=13,
        )
        bv = iree.runtime.HalBufferView(
            buffer, (1, 2), iree.runtime.HalElementType.INT_16
        )
        # NOTE: the exact bits set on type/usage/etc is implementation defined.
        self.assertEqual(
            repr(bv),
            "<HalBufferView (1, 2), element_type=0x10000010, 13 bytes (at offset 0 into 13), memory_type=DEVICE_LOCAL|HOST_VISIBLE, allowed_access=ALL, allowed_usage=TRANSFER|DISPATCH_STORAGE|MAPPING|MAPPING_PERSISTENT>",
        )
        self.assertEqual(4, bv.byte_length)

    def testBufferMap(self):
        buffer = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=13,
        )
        m = buffer.map()
        self.assertIsInstance(m, iree.runtime.MappedMemory)

    def testAllocateBufferCopy(self):
        ary = np.zeros([3, 4], dtype=np.int32) + 2
        buffer = self.allocator.allocate_buffer_copy(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            device=self.device,
            buffer=ary,
        )
        # NOTE: the exact bits set on type/usage/etc is implementation defined.
        self.assertEqual(
            repr(buffer),
            "<HalBuffer 48 bytes (at offset 0 into 48), memory_type=DEVICE_LOCAL|HOST_VISIBLE, allowed_access=ALL, allowed_usage=TRANSFER|DISPATCH_STORAGE|MAPPING|MAPPING_PERSISTENT>",
        )

    def testAllocateBufferViewCopy(self):
        ary = np.zeros([3, 4], dtype=np.int32) + 2
        buffer = self.allocator.allocate_buffer_copy(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            device=self.device,
            buffer=ary,
            element_type=iree.runtime.HalElementType.SINT_32,
        )
        # NOTE: the exact bits set on type/usage/etc is implementation defined.
        self.assertEqual(
            repr(buffer),
            "<HalBufferView (3, 4), element_type=0x20000011, 48 bytes (at offset 0 into 48), memory_type=DEVICE_LOCAL|HOST_VISIBLE, allowed_access=ALL, allowed_usage=TRANSFER|DISPATCH_STORAGE|MAPPING|MAPPING_PERSISTENT>",
        )

    def testAllocateHostStagingBufferCopy(self):
        buffer = self.allocator.allocate_host_staging_buffer_copy(
            self.device, np.int32(0)
        )
        # NOTE: the exact bits set on type/usage/etc is implementation defined.
        self.assertEqual(
            repr(buffer),
            "<HalBuffer 4 bytes (at offset 0 into 4), memory_type=DEVICE_LOCAL|HOST_VISIBLE, allowed_access=ALL, allowed_usage=TRANSFER|MAPPING|MAPPING_PERSISTENT>",
        )

    def testSemaphore(self):
        sem0 = self.device.create_semaphore(0)
        self.assertEqual(sem0.query(), 0)
        sem1 = self.device.create_semaphore(1)
        self.assertEqual(sem1.query(), 1)
        sem1.signal(2)
        self.assertEqual(sem1.query(), 2)

    def testTrivialQueueAlloc(self):
        sem = self.device.create_semaphore(0)
        buf = self.device.queue_alloca(
            1024, wait_semaphores=[(sem, 0)], signal_semaphores=[(sem, 1)]
        )
        self.assertIsInstance(buf, iree.runtime.HalBuffer)
        self.device.queue_dealloca(
            buf, wait_semaphores=[(sem, 1)], signal_semaphores=[]
        )

    def testAllocAcceptsFences(self):
        # Also tests HalFence, HalFence.insert, HalFence.wait (infinite)
        sem = self.device.create_semaphore(0)
        fence0 = iree.runtime.HalFence(1)
        fence0.insert(sem, 0)
        fence1 = iree.runtime.HalFence(1)
        fence1.insert(sem, 1)
        fence2 = iree.runtime.HalFence(2)
        fence2.insert(sem, 2)
        buf = self.device.queue_alloca(
            1024, wait_semaphores=fence0, signal_semaphores=fence1
        )
        self.assertIsInstance(buf, iree.runtime.HalBuffer)
        self.device.queue_dealloca(
            buf, wait_semaphores=fence1, signal_semaphores=fence2
        )
        self.assertTrue(fence2.wait())
        self.assertEqual(sem.query(), 2)

    def testFenceCreateAt(self):
        sem = self.device.create_semaphore(0)
        fence = iree.runtime.HalFence.create_at(sem, 1)
        self.assertFalse(fence.wait(deadline=0))
        sem.signal(1)
        self.assertTrue(fence.wait(deadline=0))

    def testSynchronousFenceFailed(self):
        sem = self.device.create_semaphore(0)
        fence = iree.runtime.HalFence.create_at(sem, 1)
        fence.fail("TEST FAILURE")
        with self.assertRaisesRegex(
            RuntimeError, "^synchronous fence failure.*TEST FAILURE"
        ):
            fence.wait(deadline=0)

    def testAsynchronousFenceFailed(self):
        sem = self.device.create_semaphore(0)
        fence = iree.runtime.HalFence.create_at(sem, 1)
        exceptions = []

        def run():
            print("SIGNALLING ASYNC FAILURE")
            time.sleep(0.2)
            fence.fail("TEST FAILURE")
            print("SIGNALLED")

        def wait():
            print("WAITING")
            try:
                fence.wait()
            except RuntimeError as e:
                exceptions.append(e)

        runner = threading.Thread(target=run)
        waiter = threading.Thread(target=wait)
        waiter.start()
        runner.start()
        waiter.join()
        runner.join()
        self.assertTrue(exceptions)
        print(exceptions)
        # Note: It is impossible to 100% guarantee that this sequences such as to
        # report an asynchronous vs synchronous failure, although we tip the odds in
        # this favor with the sleep in the signalling thread. Therefore, we do not
        # check the "asynchronous" vs "synchronous" message prefix to avoid flaky
        # test races.
        self.assertIn("TEST FAILURE", str(exceptions[0]))

    def testFenceJoin(self):
        sem1 = self.device.create_semaphore(0)
        sem2 = self.device.create_semaphore(0)
        fence1 = iree.runtime.HalFence.create_at(sem1, 1)
        fence2 = iree.runtime.HalFence.create_at(sem2, 1)
        fence = iree.runtime.HalFence.join([fence1, fence2])
        self.assertEqual(fence.timepoint_count, 2)

    def testFenceInsert(self):
        sem1 = self.device.create_semaphore(0)
        sem2 = self.device.create_semaphore(0)
        fence = iree.runtime.HalFence(2)
        fence.insert(sem1, 1)
        self.assertEqual(fence.timepoint_count, 1)
        fence.insert(sem1, 2)
        self.assertEqual(fence.timepoint_count, 1)
        fence.insert(sem2, 2)
        self.assertEqual(fence.timepoint_count, 2)

    def testFenceExtend(self):
        sem1 = self.device.create_semaphore(0)
        sem2 = self.device.create_semaphore(0)
        fence = iree.runtime.HalFence(2)
        fence.insert(sem1, 1)
        self.assertEqual(fence.timepoint_count, 1)
        fence.extend(iree.runtime.HalFence.create_at(sem2, 2))
        self.assertEqual(fence.timepoint_count, 2)

    def testRoundTripQueueCopy(self):
        original_ary = np.zeros([3, 4], dtype=np.int32) + 2
        source_bv = self.allocator.allocate_buffer_copy(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            device=self.device,
            buffer=original_ary,
            element_type=iree.runtime.HalElementType.SINT_32,
        )
        source_buffer = source_bv.get_buffer()
        target_buffer = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=source_buffer.byte_length(),
        )
        sem = self.device.create_semaphore(0)
        self.device.queue_copy(
            source_buffer,
            target_buffer,
            wait_semaphores=iree.runtime.HalFence.create_at(sem, 0),
            signal_semaphores=iree.runtime.HalFence.create_at(sem, 1),
        )
        iree.runtime.HalFence.create_at(sem, 1).wait()
        copy_ary = target_buffer.map().asarray(original_ary.shape, original_ary.dtype)
        np.testing.assert_array_equal(original_ary, copy_ary)

    def testIncompatibleSizeQueueCopy(self):
        source_buffer = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=13,
        )
        target_buffer = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=12,
        )
        sem = self.device.create_semaphore(0)
        with self.assertRaisesRegex(ValueError, "length must be less than"):
            self.device.queue_copy(
                source_buffer,
                target_buffer,
                wait_semaphores=iree.runtime.HalFence.create_at(sem, 0),
                signal_semaphores=iree.runtime.HalFence.create_at(sem, 1),
            )

    def testCommandBufferStartsByDefault(self):
        cb = iree.runtime.HalCommandBuffer(self.device)
        with self.assertRaisesRegex(RuntimeError, "FAILED_PRECONDITION"):
            cb.begin()
        cb = iree.runtime.HalCommandBuffer(self.device, begin=False)
        cb.begin()

    def testCommandBufferCopy(self):
        # Doesn't test much but that calls succeed.
        cb = iree.runtime.HalCommandBuffer(self.device)
        buffer1 = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=13,
        )
        buffer2 = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=13,
        )
        cb.copy(buffer1, buffer2, end=True)
        with self.assertRaisesRegex(RuntimeError, "FAILED_PRECONDITION"):
            cb.end()

    def testCommandBufferFill(self):
        # Doesn't test much but that calls succeed.
        cb = iree.runtime.HalCommandBuffer(self.device)
        buffer1 = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=12,
        )
        cb.fill(buffer1, np.int32(1), 0, 12, end=True)
        with self.assertRaisesRegex(RuntimeError, "FAILED_PRECONDITION"):
            cb.end()

    def testCommandBufferExecute(self):
        # Doesn't test much but that calls succeed.
        cb = iree.runtime.HalCommandBuffer(self.device)
        buffer1 = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=12,
        )
        cb.fill(buffer1, np.int32(1), 0, 12, end=True)

        sem = self.device.create_semaphore(0)
        self.device.queue_execute(
            [cb], wait_semaphores=[(sem, 0)], signal_semaphores=[(sem, 1)]
        )
        iree.runtime.HalFence.create_at(sem, 1).wait()

    def testCommandBufferExecuteAcceptsFence(self):
        # Doesn't test much but that calls succeed.
        cb = iree.runtime.HalCommandBuffer(self.device)
        buffer1 = self.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=iree.runtime.BufferUsage.DEFAULT,
            allocation_size=12,
        )
        cb.fill(buffer1, np.int32(1), 0, 12, end=True)

        sem = self.device.create_semaphore(0)
        self.device.queue_execute(
            [cb],
            wait_semaphores=iree.runtime.HalFence.create_at(sem, 0),
            signal_semaphores=iree.runtime.HalFence.create_at(sem, 1),
        )
        iree.runtime.HalFence.create_at(sem, 1).wait()


if __name__ == "__main__":
    unittest.main()
