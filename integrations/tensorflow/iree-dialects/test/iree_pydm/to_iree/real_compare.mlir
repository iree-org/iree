// RUN: iree-dialects-opt -convert-iree-pydm-to-iree %s | FileCheck  --dump-input-filter=all %s

// CHECK-LABEL: @lt
iree_pydm.func @lt(%arg0 : !iree_pydm.real, %arg1 : !iree_pydm.real) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
  // CHECK: %[[R:.*]] = arith.cmpf olt, %arg0, %arg1 : f32
  %0 = apply_compare "lt", %arg0, %arg1 : !iree_pydm.real, !iree_pydm.real
  // CHECK: return {{.*}}, %[[R]]
  return %0 : !iree_pydm.bool
}

// CHECK-LABEL: @le
iree_pydm.func @le(%arg0 : !iree_pydm.real, %arg1 : !iree_pydm.real) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
  // CHECK: %[[R:.*]] = arith.cmpf ole, %arg0, %arg1 : f32
  %0 = apply_compare "le", %arg0, %arg1 : !iree_pydm.real, !iree_pydm.real
  // CHECK: return {{.*}}, %[[R]]
  return %0 : !iree_pydm.bool
}

// CHECK-LABEL: @eq
iree_pydm.func @eq(%arg0 : !iree_pydm.real, %arg1 : !iree_pydm.real) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
  // CHECK: %[[R:.*]] = arith.cmpf oeq, %arg0, %arg1 : f32
  %0 = apply_compare "eq", %arg0, %arg1 : !iree_pydm.real, !iree_pydm.real
  // CHECK: return {{.*}}, %[[R]]
  return %0 : !iree_pydm.bool
}

// CHECK-LABEL: @is
iree_pydm.func @is(%arg0 : !iree_pydm.real, %arg1 : !iree_pydm.real) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
  // CHECK: %[[R:.*]] = arith.cmpf oeq, %arg0, %arg1 : f32
  %0 = apply_compare "is", %arg0, %arg1 : !iree_pydm.real, !iree_pydm.real
  // CHECK: return {{.*}}, %[[R]]
  return %0 : !iree_pydm.bool
}

// CHECK-LABEL: @ne
iree_pydm.func @ne(%arg0 : !iree_pydm.real, %arg1 : !iree_pydm.real) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
  // CHECK: %[[R:.*]] = arith.cmpf one, %arg0, %arg1 : f32
  %0 = apply_compare "ne", %arg0, %arg1 : !iree_pydm.real, !iree_pydm.real
  // CHECK: return {{.*}}, %[[R]]
  return %0 : !iree_pydm.bool
}

// CHECK-LABEL: @isnot
iree_pydm.func @isnot(%arg0 : !iree_pydm.real, %arg1 : !iree_pydm.real) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
  // CHECK: %[[R:.*]] = arith.cmpf one, %arg0, %arg1 : f32
  %0 = apply_compare "isnot", %arg0, %arg1 : !iree_pydm.real, !iree_pydm.real
  // CHECK: return {{.*}}, %[[R]]
  return %0 : !iree_pydm.bool
}

// CHECK-LABEL: @gt
iree_pydm.func @gt(%arg0 : !iree_pydm.real, %arg1 : !iree_pydm.real) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
  // CHECK: %[[R:.*]] = arith.cmpf ogt, %arg0, %arg1 : f32
  %0 = apply_compare "gt", %arg0, %arg1 : !iree_pydm.real, !iree_pydm.real
  // CHECK: return {{.*}}, %[[R]]
  return %0 : !iree_pydm.bool
}

// CHECK-LABEL: @ge
iree_pydm.func @ge(%arg0 : !iree_pydm.real, %arg1 : !iree_pydm.real) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
  // CHECK: %[[R:.*]] = arith.cmpf oge, %arg0, %arg1 : f32
  %0 = apply_compare "ge", %arg0, %arg1 : !iree_pydm.real, !iree_pydm.real
  // CHECK: return {{.*}}, %[[R]]
  return %0 : !iree_pydm.bool
}
