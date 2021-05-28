# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from absl import logging
from absl.testing import absltest
import iree.runtime
import numpy as np


class HalTest(absltest.TestCase):

  def testEnums(self):
    logging.info("MemoryType: %s", iree.runtime.MemoryType)
    logging.info("HOST_VISIBLE: %s", int(iree.runtime.MemoryType.HOST_VISIBLE))


if __name__ == "__main__":
  absltest.main()
