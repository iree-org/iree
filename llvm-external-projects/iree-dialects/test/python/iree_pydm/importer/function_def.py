# RUN: %PYTHON %s | iree-dialects-opt | FileCheck --enable-var-scope --dump-input-filter=all %s

from typing import List
from iree.compiler.dialects.iree_pydm.importer.test_util import *


# CHECK-LABEL: func @fully_typed_with_return
# CHECK-SAME: (%arg0: !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.integer)
# CHECK-SAME: attributes {arg_names = ["a"], cell_vars = [], free_vars = ["a"]}
# CHECK: return {{.*}} : !iree_pydm.integer
@test_import_global
def fully_typed_with_return(a: int) -> int:
  return a


# CHECK-LABEL: func @no_return
# CHECK: %[[NONE:.*]] = none
# CHECK: return %[[NONE]]
@test_import_global
def no_return():
  pass
