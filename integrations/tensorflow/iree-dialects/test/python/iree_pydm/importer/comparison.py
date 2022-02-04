# RUN: %PYTHON %s | iree-dialects-opt | FileCheck --enable-var-scope --dump-input-filter=all %s

# pytype: disable=invalid-directive
# pytype: disable=unsupported-operands

from typing import List
from iree.compiler.dialects.iree_pydm.importer.test_util import *


# CHECK-LABEL: func @binary_lt_
# CHECK-DAG: %[[L:.*]] = load_var %x
# CHECK-DAG: %[[R:.*]] = load_var %y
# CHECK: %[[LP:.*]], %[[RP:.*]] = dynamic_binary_promote %[[L]], %[[R]]
# CHECK: apply_compare "lt", %[[LP]], %[[RP]]
@test_import_global
def binary_lt_():
  x = 1
  y = 2
  return x < y


# CHECK-LABEL: func @binary_gt_
# CHECK: dynamic_binary_promote
# CHECK: apply_compare "gt"
@test_import_global
def binary_gt_():
  x = 1
  y = 2
  return x > y


# CHECK-LABEL: func @binary_lte_
# CHECK: dynamic_binary_promote
# CHECK: apply_compare "le"
@test_import_global
def binary_lte_():
  x = 1
  y = 2
  return x <= y


# CHECK-LABEL: func @binary_gte_
# CHECK: dynamic_binary_promote
# CHECK: apply_compare "ge"
@test_import_global
def binary_gte_():
  x = 1
  y = 2
  return x >= y


# CHECK-LABEL: func @binary_eq_
# CHECK: dynamic_binary_promote
# CHECK: apply_compare "eq"
@test_import_global
def binary_eq_():
  x = 1
  y = 2
  return x == y


# CHECK-LABEL: func @binary_neq_
# CHECK: dynamic_binary_promote
# CHECK: apply_compare "ne"
@test_import_global
def binary_neq_():
  x = 1
  y = 2
  return x != y


# CHECK-LABEL: func @binary_is_
# CHECK-NOT: dynamic_binary_promote
# CHECK: apply_compare "is"
@test_import_global
def binary_is_():
  x = 1
  y = 2
  return x is y


# CHECK-LABEL: func @binary_is_not_
# CHECK-NOT: dynamic_binary_promote
# CHECK: apply_compare "isnot"
@test_import_global
def binary_is_not_():
  x = 1
  y = 2
  return x is not y


# CHECK-LABEL: func @binary_in_
# CHECK-NOT: dynamic_binary_promote
# CHECK: apply_compare "in"
@test_import_global
def binary_in_():
  x = 1
  y = 2
  return x in y


# CHECK-LABEL: func @binary_not_in_
# CHECK-NOT: dynamic_binary_promote
# CHECK: apply_compare "notin"
@test_import_global
def binary_not_in_():
  x = 1
  y = 2
  return x not in y


# CHECK-LABEL: @short_circuit
# CHECK-DAG: %[[FALSE:.*]] = constant false
# CHECK-DAG: %[[X:.*]] = load_var %x
# CHECK-DAG: %[[Y:.*]] = load_var %y
# CHECK: %[[XP:.*]], %[[YP:.*]] = dynamic_binary_promote %[[X]], %[[Y]]
# CHECK: %[[R1:.*]] = apply_compare "lt", %[[XP]], %[[YP]]
# CHECK: %[[RESULT:.*]] = functional_if %[[R1]] {{.*}}{
# CHECK:   %[[Z:.*]] = load_var %z
# NOTE: Promotion happens on original loaded values, not already promoted
# values.
# CHECK:   %[[YP1:.*]], %[[ZP1:.*]] = dynamic_binary_promote %[[Y]], %[[Z]]
# CHECK:   %[[R2:.*]] = apply_compare "eq", %[[YP1]], %[[ZP1]]
# CHECK:   %[[RESULT1:.*]] = functional_if %[[R2]] {{.*}} {
# CHECK:     %[[OMEGA:.*]] = load_var %omega
# CHECK:     %[[ZP2:.*]], %[[OMEGAP2:.*]] = dynamic_binary_promote %[[Z]], %[[OMEGA]]
# CHECK:     %[[R3:.*]] = apply_compare "ge", %[[ZP2]], %[[OMEGAP2]]
# CHECK:     yield %[[R3]]
# CHECK:   } else {
# CHECK:     yield %[[FALSE]]
# CHECK:   }
# CHECK:   yield %[[RESULT1]]
# CHECK: } else {
# CHECK:   yield %[[FALSE]]
# CHECK: }
@test_import_global
def short_circuit():
  x = 1
  y = 2
  z = 3
  omega = 5
  return x < y == z >= omega


# CHECK-LABEL: nested_short_circuit_expression
# Verify that the nested expression is evaluated in the context of the if.
# CHECK: functional_if {{.*}}{
# CHECK:   apply_binary "add"
# CHECK: } else {
@test_import_global
def nested_short_circuit_expression():
  x = 1
  y = 2
  z = 3
  return x < y == (z + 6)
