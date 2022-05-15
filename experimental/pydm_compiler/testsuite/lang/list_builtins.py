# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.pydm.testing import jit


@jit
def list_len(n: int) -> int:
  """
    >>> list_len(20)
    20
  """
  lst = [1] * n
  return len(lst)
