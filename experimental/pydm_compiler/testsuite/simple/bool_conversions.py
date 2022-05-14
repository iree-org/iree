# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests that various data types coerce properly to bool.
  TODO: Returning a bool goes through the default conversion path for the VM
  and just gets returned to the user as an int.

  Int:
  >>> object_as_bool_int(0)
  0
  >>> object_as_bool_int(-1)
  1

  Bool:
  >>> object_as_bool_bool(False)
  0
  >>> object_as_bool_bool(True)
  1

  Float:
  BUG: Should return False but returns True.
  >>> object_as_bool_float(0.0)
  1
  >>> object_as_bool_float(1.0)
  1

"""

from iree.pydm.testing import jit


@jit
def object_as_bool_int(condition: int) -> bool:
  if condition:
    return True
  else:
    return False


@jit
def object_as_bool_bool(condition: bool) -> bool:
  if condition:
    return True
  else:
    return False


@jit
def object_as_bool_float(condition: float) -> bool:
  if condition:
    return True
  else:
    return False
