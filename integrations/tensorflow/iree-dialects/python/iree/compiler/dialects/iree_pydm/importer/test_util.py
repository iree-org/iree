# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import *


def test_import_global(f):
  """Imports a global function and prints corresponding IR."""
  print("// -----")
  ic = ImportContext()
  imp = Importer(ic)
  imp.import_global_function(f)
  print(ic._root_module)
  return f
