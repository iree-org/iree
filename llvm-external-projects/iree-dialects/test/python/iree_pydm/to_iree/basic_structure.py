# RUN: %PYTHON %s | iree-dialects-opt -convert-iree-pydm-to-iree | FileCheck --enable-var-scope --dump-input-filter=all %s
# This test isn't currently checking anything except that e2e lowering doesn't
# crash.

from typing import List
from mlir.dialects.iree_pydm.importer.test_util import *


# CHECK-LABEL: @return_none_no_args
@test_import_global
def return_none_no_args():
  return None


# CHECK-LABEL: @weak_integer_arg_and_return
@test_import_global
def weak_integer_arg_and_return(a: int) -> int:
  return a
