# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.pydm.testing import jit


@jit
def for_in_add_range(start: int, stop: int, step: int) -> int:
  """
    >>> for_in_add_range(0, 5, 1)
    5
  """
  v = 0
  for i in range(start, stop, step):
    v = v + 1
  return v
