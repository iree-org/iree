# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import enum
import itertools
import operator
import unittest

from iree import runtime as rt

# HAL enums whose values are bit masks that callers combine with `|` and `&`.
BITMASK_ENUMS = [
    rt.BufferCompatibility,
    rt.BufferUsage,
    rt.MemoryAccess,
    rt.MemoryType,
]


class HalEnumsTest(unittest.TestCase):
    def testBitwiseOpsMatchIntOps(self):
        for enum_type in BITMASK_ENUMS:
            members = list(enum_type.__members__.values())
            for a, b in itertools.product(members, repeat=2):
                for op in (operator.or_, operator.and_):
                    with self.subTest(
                        enum_type=enum_type.__name__,
                        a=a.name,
                        b=b.name,
                        op=op.__name__,
                    ):
                        self.assertEqual(op(a, b), op(int(a), int(b)))

    def testMemoryTypeOrKeepsBitsOfMultiBitMembers(self):
        # HOST_LOCAL has a bit that no single-bit member has. DeviceArray.to_host()
        # uses this combination for buffers that it cannot map.
        self.assertEqual(
            rt.MemoryType.HOST_LOCAL | rt.MemoryType.DEVICE_VISIBLE,
            int(rt.MemoryType.HOST_LOCAL) | int(rt.MemoryType.DEVICE_VISIBLE),
        )

    def testIndex(self):
        for enum_type in BITMASK_ENUMS:
            for member in enum_type.__members__.values():
                with self.subTest(enum_type=enum_type.__name__, member=member.name):
                    self.assertEqual(operator.index(member), int(member))

    def testFlagEnumsOnlyHaveBitsOfSingleBitMembers(self):
        # On Python 3.11.0 to 3.11.3, `|` on a nanobind flag enum (py::is_flag())
        # raises for members with a bit that no single-bit member has. Such
        # enums have to be IntEnums instead.
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
