# RUN: %PYTHON %s | iree-dialects-opt | FileCheck --enable-var-scope --dump-input-filter=all %s

from typing import List
from mlir.dialects.iree_pydm.importer.test_util import *


# CHECK-LABEL @expr_statement
# CHECK: %[[XVAL:.*]] = load_var %x
# CHECK: expr_statement_discard %[[XVAL]]
@test_import_global
def expr_statement(x: int):
  x
