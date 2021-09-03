# RUN: %PYTHON %s | iree-dialects-opt | FileCheck --enable-var-scope --dump-input-filter=all %s

from typing import List
from mlir.dialects.iree_pydm.importer.test_util import *


# CHECK-LABEL @expr_statement
# CHECK: %[[XVAL:.*]] = iree_pydm.load_free_var "x"
# CHECK: iree_pydm.expr_statement_discard %[[XVAL]]
@test_import_global
def expr_statement(x: int):
  x
