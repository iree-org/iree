# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.pydm.testing import jit


@jit
def type_error_on_return(condition: bool, true_value: int,
                         false_value: float) -> int:
  """
  Taking the false branch attempts to cast a float to an int.
    >>> type_error_on_return(0, 1, 2.0)
    Traceback (most recent call last):
    ...
    ValueError: Raised from compiled function

  Taking the true branch is legal.
    >>> type_error_on_return(1, 1, 2.0)
    1
  """
  if condition:
    return true_value
  else:
    return false_value


@jit
def unbound_local(condition: int) -> int:
  """
  >>> unbound_local(0)
  Traceback (most recent call last):
  ...
  UnboundLocalError

  BUG: This should be legal and return 1.
  >>> unbound_local(1)
  Traceback (most recent call last):
  ...
  UnboundLocalError
  """
  if condition:
    r = 1
  return r
