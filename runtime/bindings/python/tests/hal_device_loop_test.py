# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import timeit
import unittest

from iree.runtime import (
    get_device,
    HalDeviceLoopBridge,
)


class HalDeviceLoopBridgeTest(unittest.TestCase):
    def testBridge(self):
        loop = asyncio.new_event_loop()
        bridge = HalDeviceLoopBridge(self.device, loop)
        sem1 = None
        sem2 = None
        report = None

        async def main():
            def done_1(x):
                report("PYTHON: sem2.signal(1)")
                sem2.signal(1)

            def done_2(x):
                report("PYTHON: sem2.signal(2)")
                sem2.signal(2)

            f1 = bridge.on_semaphore(sem1, 1, "Semaphore 1 Signaled")
            f1.add_done_callback(done_1)
            f2 = bridge.on_semaphore(sem2, 1, "Semaphore 2 Signaled")
            f2.add_done_callback(done_2)
            f2_again = bridge.on_semaphore(sem2, 2, "Semaphore 2 Signaled Again")

            sem1.signal(1)
            f1_result = await f1
            report("PYTHON: await f1 =", f1_result)
            f2_result = await f2
            report("PYTHON: await f2 =", f2_result)
            f2_again_result = await f2_again
            report("PYTHON: await f2_again =", f2_again_result)

            self.assertEqual(f1_result, "Semaphore 1 Signaled")
            self.assertEqual(f2_result, "Semaphore 2 Signaled")
            self.assertEqual(f2_again_result, "Semaphore 2 Signaled Again")
            report("PYTHON: ASYNC MAIN() COMPLETE")

        def run_iter(with_report):
            nonlocal sem1
            nonlocal sem2
            nonlocal report
            sem1 = self.device.create_semaphore(0)
            sem2 = self.device.create_semaphore(0)
            if with_report:
                report = lambda *args: print(*args)
            else:
                report = lambda *args: None
            loop.run_until_complete(main())

        try:
            run_iter(True)
            iter_time = timeit.timeit("run_iter(False)", globals=locals(), number=10)
            print(f"Time/iter = {iter_time}s")
        finally:
            bridge.stop()

    def setUp(self):
        super().setUp()
        # TODO: Switch to local-task (experiencing some wait deadlocking
        # that needs triage).
        self.device = get_device("local-sync")
        self.allocator = self.device.allocator


if __name__ == "__main__":
    unittest.main()
