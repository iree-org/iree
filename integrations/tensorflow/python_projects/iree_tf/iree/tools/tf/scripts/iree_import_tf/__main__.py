# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import subprocess
import sys

import iree.tools.tf


def main(args=None):
  if args is None:
    args = sys.argv[1:]
  exe = iree.tools.tf.get_tool("iree-import-tf")
  return subprocess.call(args=[exe] + args)


if __name__ == "__main__":
  sys.exit(main())
