# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.pydm.testing import jit


@jit
def tuple_len() -> int:
  """
  TODO: This test isn't very strong because the compiler may be smart enough
  at some point to elide the tuple construction entirely. Should find another
  way to inject an opaque tuple that won't be optimized.
    >>> tuple_len()
    3
  """
  tpl = (1, 2, 3)
  return len(tpl)
