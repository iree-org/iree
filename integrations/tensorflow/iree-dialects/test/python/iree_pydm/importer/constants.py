# RUN: %PYTHON %s | iree-dialects-opt | FileCheck --enable-var-scope --dump-input-filter=all %s

from typing import List
from iree.compiler.dialects.iree_pydm.importer.test_util import *


# CHECK-LABEL: @const_integer
# CHECK: = constant 1 : i64 -> !iree_pydm.integer
@test_import_global
def const_integer():
  return 1


# CHECK-LABEL: @const_float
# CHECK: = constant 2.200000e+00 : f64 -> !iree_pydm.real
@test_import_global
def const_float():
  return 2.2


# CHECK-LABEL: @const_str
# CHECK: = constant "Hello" -> !iree_pydm.str
@test_import_global
def const_str():
  return "Hello"


# CHECK-LABEL: @const_bytes
# CHECK: = constant "Bonjour" -> !iree_pydm.bytes
@test_import_global
def const_bytes():
  return b"Bonjour"


# CHECK-LABEL: @const_none
# CHECK: = none
@test_import_global
def const_none():
  return None


# CHECK-LABEL: @const_true
# CHECK: = constant true -> !iree_pydm.bool
@test_import_global
def const_true():
  return True


# CHECK-LABEL: @const_false
# CHECK: = constant false -> !iree_pydm.bool
@test_import_global
def const_false():
  return False
