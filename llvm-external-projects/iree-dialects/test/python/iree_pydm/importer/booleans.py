# RUN: %PYTHON %s | iree-dialects-opt | FileCheck --enable-var-scope --dump-input-filter=all %s

from typing import List
from mlir.dialects.iree_pydm.importer.test_util import *


# CHECK-LABEL: @logical_and
# CHECK: %[[XVAL:.*]] = load_var %x
# CHECK: %[[XBOOL:.*]] = as_bool %[[XVAL]]
# CHECK: %[[R1:.*]] = functional_if %[[XBOOL]] {{.*}} {
# CHECK:   %[[YVAL:.*]] = load_var %y
# CHECK:   %[[YBOOL:.*]] = as_bool %[[YVAL]]
# CHECK:   %[[R2:.*]] = functional_if %[[YBOOL]] {{.*}} {
# CHECK:     %[[ZVAL:.*]] = load_var %z
# CHECK:     yield %[[ZVAL]]
# CHECK:   } else {
# CHECK:     yield %[[YVAL]]
# CHECK:   }
# CHECK:   yield %[[R2]]
# CHECK: } else {
# CHECK:   yield %[[XVAL]]
# CHECK: }
@test_import_global
def logical_and():
  x = 1
  y = 0
  z = 2
  return x and y and z


# # CHECK-LABEL: @logical_or
# CHECK: %[[XVAL:.*]] = load_var %x
# CHECK: %[[XBOOL:.*]] = as_bool %[[XVAL]]
# CHECK: %[[R1:.*]] = functional_if %[[XBOOL]] {{.*}} {
# CHECK:   yield %[[XVAL]]
# CHECK: } else {
# CHECK:   %[[YVAL:.*]] = load_var %y
# CHECK:   %[[YBOOL:.*]] = as_bool %[[YVAL]]
# CHECK:   %[[R2:.*]] = functional_if %[[YBOOL]] {{.*}} {
# CHECK:     yield %[[YVAL]]
# CHECK:   } else {
# CHECK:     %[[ZVAL:.*]] = load_var %z
# CHECK:     yield %[[ZVAL]]
# CHECK:   }
# CHECK:   yield %[[R2]]
# CHECK: }
@test_import_global
def logical_or():
  x = 0
  y = 1
  z = 2
  return x or y or z


# CHECK-LABEL: func @logical_not
# CHECK: %[[XVAL:.*]] = load_var %x
# CHECK: %[[XBOOL:.*]] = as_bool %[[XVAL]]
# CHECK: %[[T:.*]] = constant true
# CHECK: %[[F:.*]] = constant false
# CHECK: %[[R:.*]] = select %[[XBOOL]], %[[F]], %[[T]]
@test_import_global
def logical_not():
  x = 1
  return not x


# CHECK-LABEL: func @conditional
# CHECK: %[[XVAL:.*]] = load_var %x
# CHECK: %[[XBOOL:.*]] = as_bool %[[XVAL]]
# CHECK: %[[R1:.*]] = functional_if %[[XBOOL]] {{.*}} {
# CHECK:   %[[TWOVAL:.*]] = constant 2
# CHECK:   %[[TWOBOXED:.*]] = box %[[TWOVAL]] : !iree_pydm.integer -> !iree_pydm.object
# CHECK:   yield %[[TWOBOXED]]
# CHECK: } else {
# CHECK:   %[[THREEVAL:.*]] = constant 3
# CHECK:   %[[THREEBOXED:.*]] = box %[[THREEVAL]] : !iree_pydm.integer -> !iree_pydm.object
# CHECK:   yield %[[THREEBOXED]]
# CHECK: }
@test_import_global
def conditional():
  x = 1
  return 2 if x else 3
