# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.pydm.testing import jit


@jit
def getitem_int(a: int, b: int) -> int:
  """
    >>> getitem_int(1, 7)
    7
  """
  lst = [a, b, 9]
  return lst[1]

print(getitem_int(1, 7))
