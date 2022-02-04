// RUN: iree-dialects-opt -split-input-file -convert-iree-pydm-to-iree %s | FileCheck  --dump-input-filter=all %s

// CHECK-LABEL: @none_constant
iree_pydm.func @none_constant() -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK: %[[CST0:.*]] = arith.constant 0 : i32
  // CHECK: %[[CST1:.*]] = arith.constant 0 : i32
  // CHECK: return %[[CST1]], %[[CST0]]
  %0 = none
  return %0 : !iree_pydm.none
}

// CHECK-LABEL: @constant_integer_trunc
iree_pydm.func @constant_integer_trunc() -> (!iree_pydm.exception_result, !iree_pydm.integer) {
  // CHECK: arith.constant -10 : i32
  %0 = constant -10 : i64 -> !iree_pydm.integer
  return %0 : !iree_pydm.integer
}

// CHECK-LABEL: @constant_real_trunc
iree_pydm.func @constant_real_trunc() -> (!iree_pydm.exception_result, !iree_pydm.real) {
  // CHECK: arith.constant -2.000000e+00 : f32
  %0 = constant -2.0 : f64 -> !iree_pydm.real
  return %0 : !iree_pydm.real
}
