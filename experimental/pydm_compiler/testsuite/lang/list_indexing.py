# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.pydm.testing import jit


@jit
def construct_and_index_const(a: int, b: int) -> int:
  """
    >>> construct_and_index_const(1, 7)
    7
  """
  lst = [a, b, 9]
  return lst[1]


@jit
def construct_and_index_var_int(a: int, b: int, idx: int) -> int:
  """
  Various indexing into a constructed list of [1, 7, 9]

  Negative indexing:
    >>> construct_and_index_var_int(1, 7, -3)
    1
    >>> construct_and_index_var_int(1, 7, -2)
    7
    >>> construct_and_index_var_int(1, 7, -1)
    9

  Positive indexing:
    >>> construct_and_index_var_int(1, 7, 0)
    1
    >>> construct_and_index_var_int(1, 7, 1)
    7
    >>> construct_and_index_var_int(1, 7, 2)
    9

  Out of bound indexing:
    >>> construct_and_index_var_int(1, 7, 4)
    Traceback (most recent call last):
    ...
    IndexError: Raised from compiled function

    >>> construct_and_index_var_int(1, 7, -4)
    Traceback (most recent call last):
    ...
    IndexError: Raised from compiled function
  """
  lst = [a, b, 9]
  return lst[idx]


@jit
def list_multiply(count: int, index: int) -> int:
  """
  Note: The ABI doesn't yet us return a list yet, so we reconstruct it.
    >>> full_list = [list_multiply(5, i) for i in range(15)]
    >>> full_list
    [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]

  Out of bound situations:
    >>> list_multiply(5, 15)
    Traceback (most recent call last):
    ...
    IndexError: Raised from compiled function
    >>> list_multiply(0, 0)
    Traceback (most recent call last):
    ...
    IndexError: Raised from compiled function
    >>> list_multiply(-1, 0)
    Traceback (most recent call last):
    ...
    IndexError: Raised from compiled function
  """
  lst = [1, 2, 3] * count
  return lst[index]


@jit
def list_setitem(write: int, value: int, read: int) -> int:
  """
  Write positive index, read negative index:
    >>> list_setitem(1, 99, -2)
    99

  Write negative index, read positive index:
    >>> list_setitem(-2, 100, 1)
    100

  Out of bound situations:
    >>> list_setitem(3, 100, 0)
    Traceback (most recent call last):
    ...
    IndexError: Raised from compiled function
    >>> list_setitem(-4, 100, 0)
    Traceback (most recent call last):
    ...
    IndexError: Raised from compiled function
  """
  lst = [1, 2, 3]
  lst[write] = value
  return lst[read]
