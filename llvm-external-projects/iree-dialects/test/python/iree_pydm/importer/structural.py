# RUN: %PYTHON %s | iree-dialects-opt | FileCheck --enable-var-scope --dump-input-filter=all %s

from typing import List
from iree.compiler.dialects.iree_pydm.importer.test_util import *


# CHECK-LABEL @expr_statement
# CHECK: %[[XVAL:.*]] = load_var %x
# CHECK: expr_statement_discard %[[XVAL]]
@test_import_global
def expr_statement(x: int):
  x


# CHECK-LABEL @make_tuple
@test_import_global
def make_tuple(x, y) -> tuple:
  # CHECK: %[[X:.*]] = load_var %x
  # CHECK: %[[Y:.*]] = load_var %y
  # CHECK: %[[RESULT:.*]] = make_tuple %[[X]], %[[Y]]
  # CHECK: return %[[RESULT]]
  return x, y
