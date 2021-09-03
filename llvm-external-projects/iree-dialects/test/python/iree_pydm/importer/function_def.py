# RUN: %PYTHON %s | iree-dialects-opt | FileCheck --enable-var-scope --dump-input-filter=all %s

from typing import List
from mlir.dialects.iree_pydm.importer.test_util import *


# CHECK-LABEL: iree_pydm.func @fully_typed_with_return
# CHECK-SAME: (%arg0: !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.integer)
# CHECK-SAME: attributes {arg_names = ["a"], cell_vars = [], free_vars = ["a"]}
# CHECK: iree_pydm.return {{.*}} : !iree_pydm.integer
@test_import_global
def fully_typed_with_return(a: int) -> int:
  return a


# CHECK-LABEL: iree_pydm.func @no_return
# CHECK: %[[NONE:.*]] = iree_pydm.none
# CHECK: iree_pydm.return %[[NONE]]
@test_import_global
def no_return():
  pass
