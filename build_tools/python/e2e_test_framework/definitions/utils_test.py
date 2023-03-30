## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from e2e_test_framework.definitions import utils


class UtilsTest(unittest.TestCase):

  def test_transform_flags(self):
    flags = utils.transform_flags(
        flags=[
            r"${HOLDER_A} ${HOLDER_B}", r"--key=${HOLDER_A}", "--no-value-key",
            r"--filter=x=${HOLDER_A}"
        ],
        map_funcs=[
            lambda value: value.replace(r"${HOLDER_A}", "val_a"),
            lambda value: value.replace(r"${HOLDER_B}", "val_b")
        ])

    self.assertEquals(
        flags,
        ["val_a val_b", "--key=val_a", "--no-value-key", "--filter=x=val_a"])


if __name__ == "__main__":
  unittest.main()
