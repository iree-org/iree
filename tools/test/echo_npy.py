# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy
import os
import sys

with open(os.path.realpath(sys.argv[1]), 'rb') as f:
  f.seek(0, 2)
  file_len = f.tell()
  f.seek(0, 0)
  while f.tell() < file_len:
    print(numpy.load(f))
