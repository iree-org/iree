# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

from iree.compiler.transforms import ireec


class CompilerTest(unittest.TestCase):

  def testDefaultOptions(self):
    options = ireec.CompilerOptions()
    self.assertEqual(repr(options), "<CompilerOptions:[]>")

  def testOptionsBadArg(self):
    with self.assertRaisesRegex(ValueError, "option not found: foobar"):
      options = ireec.CompilerOptions("--foobar")

  def testOptionsBoolArgImplicit(self):
    options = ireec.CompilerOptions("--iree-tflite-bindings-support")
    self.assertEqual(
        repr(options),
        "<CompilerOptions:['--iree-tflite-bindings-support=true']>")

  def testOptionsBoolArgExplicit(self):
    options = ireec.CompilerOptions("--iree-tflite-bindings-support=true")
    self.assertEqual(
        repr(options),
        "<CompilerOptions:['--iree-tflite-bindings-support=true']>")

  def testOptionsEnumArg(self):
    options = ireec.CompilerOptions("--iree-input-type=mhlo")
    self.assertEqual(repr(options),
                     "<CompilerOptions:['--iree-input-type=mhlo']>")

  def testListOption(self):
    options = ireec.CompilerOptions("--iree-hal-target-backends=llvm-cpu,vmvx")
    self.assertEqual(
        repr(options),
        "<CompilerOptions:['--iree-hal-target-backends=llvm-cpu,vmvx']>")
    print(options)

  def testMultipleOptions(self):
    options = ireec.CompilerOptions("--iree-input-type=mhlo",
                                    "--iree-tflite-bindings-support=true")
    self.assertEqual(
        repr(options),
        "<CompilerOptions:['--iree-tflite-bindings-support=true', '--iree-input-type=mhlo']>"
    )


if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)
  unittest.main()
