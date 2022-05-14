# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.pydm.testing import jit


@jit
def add_int_args(a: int, b: int) -> int:
  """
  >>> add_int_args(3, 4)
  7
  """
  return a + b


@jit
def sub_int_args(a: int, b: int) -> int:
  """
  >>> sub_int_args(3, 4)
  -1
  """
  return a - b


@jit
def int_arithmetic_expr(a: int, b: int, c: int) -> int:
  """
  >>> int_arithmetic_expr(5, 4, 10)
  103
  >>> int_arithmetic_expr(4, 4, 10)
  40
  """
  if a - b:
    return 3 * a + 22 * b
  else:
    return 4 * c


@jit
def float_arithmetic_expr(a: float, b: float) -> float:
  """
  BUG: Floating point arithmetic being mishandled.
  >>> float_arithmetic_expr(3.0, 5.0)
  0.0
  """
  return 3.0 * a - 2.0 * b
