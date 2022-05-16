# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.pydm.testing import jit


@jit
def len_range_stop_only(n: int) -> int:
  """
    >>> len_range_stop_only(20)
    20
    >>> len_range_stop_only(0)
    0
    >>> len_range_stop_only(-1)
    0
  """
  return len(range(n))


@jit
def len_range_start_stop(start: int, stop: int) -> int:
  """
    >>> len_range_start_stop(0, 20)
    20
    >>> len_range_start_stop(20, 20)
    0
    >>> len_range_start_stop(21, 20)
    0
    >>> len_range_start_stop(-20, 20)
    40
    >>> len_range_start_stop(0, 0)
    0
  """
  return len(range(start, stop))


@jit
def len_range_start_stop_step(start: int, stop: int, step: int) -> int:
  """
    >>> len_range_start_stop_step(0, 20, 1)
    20
    >>> len_range_start_stop_step(0, 20, 2)
    10
    >>> len_range_start_stop_step(0, 20, 3)
    7
    >>> len_range_start_stop_step(20, -20, -1)
    40
    >>> len_range_start_stop_step(20, -20, -3)
    14
  """
  return len(range(start, stop, step))


@jit
def range_step_0_error(step: int) -> int:
  """
    >>> range_step_0_error(0)
    Traceback (most recent call last):
    ...
    ValueError: Raised from compiled function
  """
  return len(range(0, 20, step))
