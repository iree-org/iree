# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Tuple

from io import StringIO
import unittest
import sys

from iree.runtime.scripts.iree_devices import __main__ as cli


def run_cli(*args) -> Tuple[int, str, str]:
  capture_stdout = StringIO()
  capture_stderr = StringIO()
  sys.stdout = capture_stdout
  sys.stderr = capture_stderr
  try:
    rc = cli.main(args)
  finally:
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
  return rc, capture_stdout.getvalue(), capture_stderr.getvalue()


class DevicesCliTest(unittest.TestCase):

  def testLs(self):
    rc, output, err = run_cli("ls")
    self.assertEqual(rc, 0)
    self.assertIn("vmvx:0\tdefault", output)

  def testLsTryCreate(self):
    rc, output, err = run_cli("ls", "--try-create")
    self.assertEqual(rc, 0)
    self.assertIn("vmvx:0\tdefault\tSUCCESS", output)

  def testLsTryCreateExplicitDriver(self):
    rc, output, err = run_cli("ls", "--try-create", "-d", "vmvx")
    self.assertEqual(rc, 0)
    self.assertIn("vmvx:0\tdefault\tSUCCESS", output)

  def testLsTryCreateExplicitDriverNotFound(self):
    rc, output, err = run_cli("ls", "--try-create", "-d", "DOES_NOT_EXIST")
    self.assertEqual(rc, 0)
    self.assertIn("Could not create driver DOES_NOT_EXIST", err)

  def testTestIndexedDevice(self):
    rc, output, err = run_cli("test", "vmvx:0")
    self.assertEqual(rc, 0)
    self.assertIn("Creating device vmvx:0... SUCCESS", output)

  def testTestDefaultDevice(self):
    rc, output, err = run_cli("test", "vmvx")
    self.assertEqual(rc, 0)
    self.assertIn("Creating device vmvx... SUCCESS", output)

  def testTestNonExisting(self):
    rc, output, err = run_cli("test", "NOT_EXISTING")
    self.assertEqual(rc, 1)
    self.assertIn("Creating device NOT_EXISTING... ERROR", output)


if __name__ == "__main__":
  unittest.main()
