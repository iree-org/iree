# RUN: %PYTHON %s | iree-dialects-opt | FileCheck --enable-var-scope --dump-input-filter=all %s

from typing import List
from mlir.dialects.iree_pydm.importer.test_util import *


# CHECK-LABEL: @simple_if
# CHECK: %[[COND:.*]] = load_var %cond
# CHECK: %[[COND_BOOL:.*]] = as_bool %[[COND]]
# CHECK: %[[COND_PRED:.*]] = bool_to_pred %[[COND_BOOL]]
# CHECK: cond_br %2, ^bb1, ^bb2
# CHECK: ^bb1:
# CHECK: %[[A:.*]] = load_var %a
# CHECK: return %[[A]]
# CHECK: ^bb2:
# CHECK: %[[B:.*]] = load_var %b
# CHECK: return %[[B]]
@test_import_global
def simple_if(cond, a, b):
  if cond:
    return a
  else:
    return b


# CHECK-LABEL: @if_fallthrough
# CHECK: cond_br {{.*}}, ^bb1, ^bb2
# CHECK: ^bb1:
# CHECK: br ^bb3
# CHECK: ^bb2:
# CHECK: br ^bb3
# CHECK: ^bb3:
# CHECK: return
@test_import_global
def if_fallthrough(cond, a, b):
  if cond:
    c = a
  else:
    c = b
  return c


# CHECK-LABEL: @if_noelse
# CHECK: cond_br {{.*}}, ^bb1, ^bb2
# CHECK: ^bb1:
# CHECK: br ^bb2
# CHECK: ^bb2:
# CHECK: return
@test_import_global
def if_noelse(cond, a, b):
  c = 1
  if cond:
    c = a
  return c


# CHECK-LABEL: @if_elif
# CHECK: cond_br {{.*}}, ^bb1, ^bb2
# CHECK: ^bb1:
# CHECK: br ^bb6
# CHECK: ^bb2:
# CHECK: cond_br {{.*}}, ^bb3, ^bb4
# CHECK: ^bb3:
# CHECK: br ^bb5
# CHECK: ^bb4:
# CHECK: br ^bb5
# CHECK: ^bb5:
# CHECK: br ^bb6
# CHECK: ^bb6:
# CHECK: return
@test_import_global
def if_elif(cond, a, b):
  if cond:
    c = a
  elif b:
    c = 2
  else:
    c = 3
  return c
