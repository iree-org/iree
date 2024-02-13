# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import gc
import threading
import unittest

from iree.runtime import (
    get_device,
    HalDeviceLoop,
)


class HalDeviceLoopTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.device = get_device("local-task")
        self.allocator = self.device.allocator
        self.loop = HalDeviceLoop(self.device)
        self.loop_thread = threading.Thread(target=self.loop.run)
        self.loop_thread.start()

    def tearDown(self):
        print("(((SIGNAL SHUTDOWN)))")
        self.loop.signal_shutdown()
        print("(((JOINING)))")
        self.loop_thread.join()
        print("(((JOINED)))")

    def testFuture(self):
        sem1 = self.device.create_semaphore(0)
        sem2 = self.device.create_semaphore(0)

        async def main():
            print("One fish")
            f1 = asyncio.Future()
            f1.add_done_callback(lambda x: print("FUTURE1: DONE"))
            f2 = asyncio.Future()
            f2.add_done_callback(lambda x: print("FUTURE2: DONE"))
            await asyncio.sleep(0.5)
            self.loop.on_semaphore(sem1, 1, f1)
            await asyncio.sleep(0.5)
            self.loop.on_semaphore(sem2, 1, f2)
            await asyncio.sleep(0.5)
            sem1.signal(1)
            await asyncio.sleep(0.5)
            sem2.signal(1)

            print("FUTURE1:", await f1)
            print("FUTURE2:", await f2)

        asyncio.run(main())
        assert False


if __name__ == "__main__":
    unittest.main()
