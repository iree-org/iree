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
    10
    >>> for_in_add_range(1, 6, 1)
    15
    >>> for_in_add_range(6, -2, -1)
    20
    >>> for_in_add_range(2, 8, 3)
    7
    >>> for_in_add_range(6, -2, -3)
    9
  """
  v = 0
  for i in range(start, stop, step):
    v = v + i
  return v


@jit
def for_in_count_range(start: int, stop: int, step: int) -> int:
  """
    >>> for_in_count_range(0, 5, 1)
    5
    >>> for_in_count_range(1, 6, 1)
    5
    >>> for_in_count_range(6, -2, -1)
    8
    >>> for_in_count_range(2, 8, 3)
    2
    >>> for_in_count_range(6, -2, -3)
    3
  """
  v = 0
  for i in range(start, stop, step):
    v = v + 1
  return v
