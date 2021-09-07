# RUN: %PYTHON %s | iree-dialects-opt | FileCheck --enable-var-scope --dump-input-filter=all %s

from typing import List
from mlir.dialects.iree_pydm.importer.test_util import *


# CHECK-LABEL: @binary_add
# CHECK: %[[L:.*]] = load_var %a
# CHECK: %[[R:.*]] = load_var %b
# CHECK: %[[LP:.*]], %[[RP:.*]] = dynamic_binary_promote %[[L]], %[[R]]
# CHECK: apply_binary "add", %[[LP]], %[[RP]]
@test_import_global
def binary_add(a, b):
  return a + b


# CHECK-LABEL: @binary_sub
# CHECK: apply_binary "sub"
@test_import_global
def binary_sub(a, b):
  return a - b


# CHECK-LABEL: @binary_mul
# CHECK: apply_binary "mul"
@test_import_global
def binary_mul(a, b):
  return a * b


# CHECK-LABEL: @binary_matmul
# CHECK: apply_binary "matmul"
@test_import_global
def binary_matmul(a, b):
  return a @ b


# CHECK-LABEL: @binary_truediv
# CHECK: apply_binary "truediv"
@test_import_global
def binary_truediv(a, b):
  return a / b


# CHECK-LABEL: @binary_floordiv
# CHECK: apply_binary "floordiv"
@test_import_global
def binary_floordiv(a, b):
  return a // b


# CHECK-LABEL: @binary_mod
# CHECK: apply_binary "mod"
@test_import_global
def binary_mod(a, b):
  return a % b


# CHECK-LABEL: @binary_pow
# CHECK: apply_binary "pow"
@test_import_global
def binary_pow(a, b):
  return a**b


# CHECK-LABEL: @binary_lshift
# CHECK: apply_binary "lshift"
@test_import_global
def binary_lshift(a, b):
  return a << b


# CHECK-LABEL: @binary_rshift
# CHECK: apply_binary "rshift"
@test_import_global
def binary_rshift(a, b):
  return a >> b


# CHECK-LABEL: @binary_and
# CHECK: apply_binary "and"
@test_import_global
def binary_and(a, b):
  return a & b


# CHECK-LABEL: @binary_or
# CHECK: apply_binary "or"
@test_import_global
def binary_or(a, b):
  return a | b


# CHECK-LABEL: @binary_xor
# CHECK: apply_binary "xor"
@test_import_global
def binary_xor(a, b):
  return a ^ b
