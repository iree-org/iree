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
    def findEnums(self, base):
        enums = [
            value
            for value in vars(rt._binding).values()
            if isinstance(value, enum.EnumMeta) and issubclass(value, base)
        ]
        self.assertTrue(enums, f"found no enums of type {base.__name__}")
        return enums

    def testBitwiseOps(self):
        for enum_type in self.findEnums(int):
            members = enum_type.__members__.values()
            for a in members:
                for b in members:
                    self.assertEqual(a | b, int(a) | int(b))
                    self.assertEqual(a & b, int(a) & int(b))

    def testIndex(self):
        for enum_type in self.findEnums(int):
            for member in enum_type.__members__.values():
                self.assertEqual(operator.index(member), int(member))

    def testFlagEnumsOnlyHaveBitsOfSingleBitMembers(self):
        for enum_type in self.findEnums(enum.Flag):
            single_bits = 0
            multi_bits = 0
            for member in enum_type.__members__.values():
                value = int(member)
                if value & (value - 1):
                    multi_bits |= value
                else:
                    single_bits |= value
            self.assertEqual(multi_bits & ~single_bits, 0, enum_type.__name__)


if __name__ == "__main__":
    unittest.main()
