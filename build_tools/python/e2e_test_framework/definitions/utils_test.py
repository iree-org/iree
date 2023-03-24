## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from e2e_test_framework.definitions import utils


class UtilsTest(unittest.TestCase):

  def test_materialize_flags(self):
    flags = utils.materialize_flags(
        flags=[
            r"${HOLDER_A} ${HOLDER_B}", r"--key=${HOLDER_A}", "--no-value-key"
        ],
        map_funcs=[
            lambda value: value.replace(r"${HOLDER_A}", "val_a"),
            lambda value: value.replace(r"${HOLDER_B}", "val_b")
        ])

    self.assertEquals(flags, ["val_a val_b", "--key=val_a", "--no-value-key"])

  def test_materialize_flags_iterative_substitution(self):
    flags = utils.materialize_flags(
        flags=[r"--key=${HOLDER_A}"],
        map_funcs=[
            lambda value: value.replace(r"${HOLDER_B}", "b"),
            lambda value: value.replace(r"${HOLDER_A}", r"val_${HOLDER_B}")
        ])

    self.assertEquals(flags, ["--key=val_b"])

  def test_materialize_flags_too_many_iterations(self):
    map_funcs = [
        lambda value: value.replace(r"${HOLDER_A}", r"val_${HOLDER_A}")
    ]

    self.assertRaises(
        ValueError, lambda: utils.materialize_flags(flags=[r"${HOLDER_A}"],
                                                    map_funcs=map_funcs))


if __name__ == "__main__":
  unittest.main()
