# RUN: %PYTHON %s | iree-dialects-opt | FileCheck --enable-var-scope --dump-input-filter=all %s

from typing import List
from mlir.dialects.iree_pydm.importer.test_util import *


# CHECK-LABEL: @logical_and
# CHECK: %[[XVAL:.*]] = iree_pydm.load_free_var "x"
# CHECK: %[[XBOOL:.*]] = iree_pydm.as_bool %[[XVAL]]
# CHECK: %[[XPRED:.*]] = iree_pydm.bool_to_pred %[[XBOOL]]
# CHECK: %[[R1:.*]] = scf.if %[[XPRED]] {{.*}} {
# CHECK:   %[[YVAL:.*]] = iree_pydm.load_free_var "y"
# CHECK:   %[[YBOOL:.*]] = iree_pydm.as_bool %[[YVAL]]
# CHECK:   %[[YPRED:.*]] = iree_pydm.bool_to_pred %[[YBOOL]]
# CHECK:   %[[R2:.*]] = scf.if %[[YPRED]] {{.*}} {
# CHECK:     %[[ZVAL:.*]] = iree_pydm.load_free_var "z"
# CHECK:     scf.yield %[[ZVAL]]
# CHECK:   } else {
# CHECK:     scf.yield %[[YVAL]]
# CHECK:   }
# CHECK:   scf.yield %[[R2]]
# CHECK: } else {
# CHECK:   scf.yield %[[XVAL]]
# CHECK: }
@test_import_global
def logical_and():
  x = 1
  y = 0
  z = 2
  return x and y and z


# # CHECK-LABEL: @logical_or
# CHECK: %[[XVAL:.*]] = iree_pydm.load_free_var "x"
# CHECK: %[[XBOOL:.*]] = iree_pydm.as_bool %[[XVAL]]
# CHECK: %[[XPRED:.*]] = iree_pydm.bool_to_pred %[[XBOOL]]
# CHECK: %[[R1:.*]] = scf.if %[[XPRED]] {{.*}} {
# CHECK:   scf.yield %[[XVAL]]
# CHECK: } else {
# CHECK:   %[[YVAL:.*]] = iree_pydm.load_free_var "y"
# CHECK:   %[[YBOOL:.*]] = iree_pydm.as_bool %[[YVAL]]
# CHECK:   %[[YPRED:.*]] = iree_pydm.bool_to_pred %[[YBOOL]]
# CHECK:   %[[R2:.*]] = scf.if %[[YPRED]] {{.*}} {
# CHECK:     scf.yield %[[YVAL]]
# CHECK:   } else {
# CHECK:     %[[ZVAL:.*]] = iree_pydm.load_free_var "z"
# CHECK:     scf.yield %[[ZVAL]]
# CHECK:   }
# CHECK:   scf.yield %[[R2]]
# CHECK: }
@test_import_global
def logical_or():
  x = 0
  y = 1
  z = 2
  return x or y or z


# CHECK-LABEL: func @logical_not
# CHECK: %[[XVAL:.*]] = iree_pydm.load_free_var "x"
# CHECK: %[[XBOOL:.*]] = iree_pydm.as_bool %[[XVAL]]
# CHECK: %[[T:.*]] = iree_pydm.constant true
# CHECK: %[[F:.*]] = iree_pydm.constant false
# CHECK: %[[R:.*]] = iree_pydm.select %[[XBOOL]], %[[F]], %[[T]]
@test_import_global
def logical_not():
  x = 1
  return not x


# CHECK-LABEL: func @conditional
# CHECK: %[[XVAL:.*]] = iree_pydm.load_free_var "x"
# CHECK: %[[XBOOL:.*]] = iree_pydm.as_bool %[[XVAL]]
# CHECK: %[[XPRED:.*]] = iree_pydm.bool_to_pred %[[XBOOL]]
# CHECK: %[[R1:.*]] = scf.if %[[XPRED]] {{.*}} {
# CHECK:   %[[TWOVAL:.*]] = iree_pydm.constant 2
# CHECK:   %[[TWOBOXED:.*]] = iree_pydm.box %[[TWOVAL]] : !iree_pydm.integer -> !iree_pydm.object
# CHECK:   scf.yield %[[TWOBOXED]]
# CHECK: } else {
# CHECK:   %[[THREEVAL:.*]] = iree_pydm.constant 3
# CHECK:   %[[THREEBOXED:.*]] = iree_pydm.box %[[THREEVAL]] : !iree_pydm.integer -> !iree_pydm.object
# CHECK:   scf.yield %[[THREEBOXED]]
# CHECK: }
@test_import_global
def conditional():
  x = 1
  return 2 if x else 3
