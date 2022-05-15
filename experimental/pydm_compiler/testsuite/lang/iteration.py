# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.pydm.testing import jit


@jit
def for_in(n: int) -> int:
  """
    >>> for_in(2)
    20
  """
  lst = [5, 1, 3] * n
  v = 0
  for i in lst:
    v = v + i
  return v
