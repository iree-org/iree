# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree import runtime as rt
import numpy as np
import unittest


class FlagsTest(unittest.TestCase):

  def testParse(self):
    # We always have the logging verbose level available so use it.
    rt.flags.parse_flags("--iree_v=1")

  def testParseError(self):
    with self.assertRaisesRegex(ValueError, "flag 'barbar' not recognized"):
      rt.flags.parse_flags("--barbar")


if __name__ == "__main__":
  unittest.main()
