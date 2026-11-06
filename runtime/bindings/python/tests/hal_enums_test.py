# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import enum
import operator
import unittest

from iree import runtime as rt


class HalEnumsTest(unittest.TestCase):
    def testOr(self):
        host_local = rt.MemoryType.HOST_LOCAL
        device_visible = rt.MemoryType.DEVICE_VISIBLE
        self.assertEqual(
            host_local | device_visible, int(host_local) | int(device_visible)
        )

    def testIndex(self):
        host_local = rt.MemoryType.HOST_LOCAL
        self.assertEqual(operator.index(host_local), int(host_local))

    def testFlagEnumsOnlyHaveBitsOfSingleBitMembers(self):
        flag_enums = [
            value
            for value in vars(rt._binding).values()
            if isinstance(value, type) and issubclass(value, enum.Flag)
        ]
        self.assertTrue(flag_enums, "found no flag enums to check")
        for enum_type in flag_enums:
            single_bits = 0
            multi_bits = 0
            for member in enum_type.__members__.values():
                value = int(member)
                if value & (value - 1):
                    multi_bits |= value
                else:
                    single_bits |= value
            with self.subTest(enum_type=enum_type.__name__):
                self.assertEqual(multi_bits & ~single_bits, 0)


if __name__ == "__main__":
    unittest.main()
