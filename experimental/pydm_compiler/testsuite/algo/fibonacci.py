# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.pydm.testing import jit, SimpleModule

# Due to restrictions in the way that simple_lang is implemented, recursive
# modules need to be defined at the top level like this.
FibRecursive = SimpleModule()


@FibRecursive.export_pyfunc
def fib_recursive(n: int) -> int:
  """
  BUG: Not legalizing:
  error: failed to legalize operation 'iree_pydm.dynamic_call' that was explicitly marked illegal
  >>> FibRecursive.exports.fib_recursive(5)
  Traceback (most recent call last):
  ...
  RuntimeError: Failure while executing pass pipeline.
  """
  if n <= 1:
    return n
  return fib_recursive(n - 1) + fib_recursive(n - 2)


@jit
def fib_list_item(n: int) -> int:
  """
  >>> fib_list_item(20)
  6765
  """
  values = [0] * (n + 1)
  values[0] = 0
  values[1] = 1
  for i in range(2, n + 1):
    values[i] = values[i - 1] + values[i - 2]
  return values[n]


@jit
def fib_spaceopt_item(n: int) -> int:
  """
  >>> fib_spaceopt_item(20)
  6765
  """
  a = 0
  b = 1
  if n == 0:
    return a
  # TODO: Switch to for..in when crash is fixed.
  i = 2
  while i <= n:  # TODO: Upgrade to for...range
    c = a + b
    a = b
    b = c
    i = i + 1  # TODO: Support AugAssign
  # for i in range(n):
  #   c = a + b
  #   a = b
  #   b = c

  return b
