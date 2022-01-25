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


# CHECK-LABEL @literal_list
@test_import_global
def literal_list(x, y) -> list:
  # CHECK: %[[X:.*]] = load_var %x
  # CHECK: %[[Y:.*]] = load_var %y
  # CHECK: %[[RESULT:.*]] = make_list %[[X]], %[[Y]]
  # CHECK: %[[BOXED:.*]] = box %[[RESULT]] : !iree_pydm.list -> <!iree_pydm.list>
  # CHECK: return %[[BOXED]]
  return [x, y]


# CHECK-LABEL @pass_statement
@test_import_global
def pass_statement():
  pass


# CHECK-LABEL @subscript
@test_import_global
def subscript(lst: list, index: int):
  # CHECK: %[[LIST:.*]] = load_var %lst : !iree_pydm.free_var_ref -> !iree_pydm.object
  # CHECK: %[[INDEX:.*]] = load_var %index : !iree_pydm.free_var_ref -> !iree_pydm.object
  # CHECK: %exc_result, %result = subscript %[[LIST]][%[[INDEX]]] : !iree_pydm.object, !iree_pydm.object -> !iree_pydm.object
  # CHECK: raise_on_failure %exc_result : !iree_pydm.exception_result
  return lst[index]
