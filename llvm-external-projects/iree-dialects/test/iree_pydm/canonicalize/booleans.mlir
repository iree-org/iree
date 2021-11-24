// RUN: iree-dialects-opt -split-input-file --allow-unregistered-dialect -canonicalize %s | FileCheck  --dump-input-filter=all %s

// CHECK-LABEL: @fold_none
iree_pydm.func @fold_none(%arg0 : !iree_pydm.none) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
  // CHECK: %[[F:.*]] = constant false -> !iree_pydm.bool
  // CHECK: return %[[F]]
  %0 = as_bool %arg0 : !iree_pydm.none -> !iree_pydm.bool
  return %0 : !iree_pydm.bool
}

// -----
// CHECK-LABEL: @elide_as_bool_from_bool
iree_pydm.func @elide_as_bool_from_bool(%arg0 : !iree_pydm.bool) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
  // CHECK: return %arg0
  %0 = as_bool %arg0 : !iree_pydm.bool -> !iree_pydm.bool
  return %0 : !iree_pydm.bool
}

// -----
// CHECK-LABEL: @as_bool_from_integer_to_compare
iree_pydm.func @as_bool_from_integer_to_compare(%arg0 : !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
  // CHECK-DAG: %[[Z:.*]] = constant 0 : i64 -> !iree_pydm.integer
  // CHECK-DAG: %[[T:.*]] = constant true -> !iree_pydm.bool
  // CHECK-DAG: %[[F:.*]] = constant false -> !iree_pydm.bool
  // CHECK: %[[CMP:.*]] = apply_compare "eq", %arg0, %[[Z]]
  // CHECK: %[[SEL:.*]] = select %[[CMP]], %[[F]], %[[T]] : !iree_pydm.bool
  // CHECK: return %[[SEL]]
  %0 = as_bool %arg0 : !iree_pydm.integer -> !iree_pydm.bool
  return %0 : !iree_pydm.bool
}

// -----
// CHECK-LABEL: @as_bool_from_real_to_compare
iree_pydm.func @as_bool_from_real_to_compare(%arg0 : !iree_pydm.real) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
  // CHECK-DAG: %[[Z:.*]] = constant 0.000000e+00 : f64 -> !iree_pydm.real
  // CHECK-DAG: %[[T:.*]] = constant true -> !iree_pydm.bool
  // CHECK-DAG: %[[F:.*]] = constant false -> !iree_pydm.bool
  // CHECK: %[[CMP:.*]] = apply_compare "eq", %arg0, %[[Z]]
  // CHECK: %[[SEL:.*]] = select %[[CMP]], %[[F]], %[[T]] : !iree_pydm.bool
  // CHECK: return %[[SEL]]
  %0 = as_bool %arg0 : !iree_pydm.real -> !iree_pydm.bool
  return %0 : !iree_pydm.bool
}

// -----
// CHECK-LABEL: @fold_bool_to_pred_from_constant
iree_pydm.func @fold_bool_to_pred_from_constant() -> (!iree_pydm.exception_result, !iree_pydm.none) {
  %0 = iree_pydm.constant true -> !iree_pydm.bool
  // CHECK: %[[P:.*]] = constant true -> i1
  // CHECK: "custom.donotoptimize"(%[[P]])
  %1 = bool_to_pred %0
  "custom.donotoptimize"(%1) : (i1) -> ()
  %none = none
  return %none : !iree_pydm.none
}

// -----
// CHECK-LABEL: @as_bool_from_real_to_compare
iree_pydm.func @as_bool_from_real_to_compare(%arg0: !iree_pydm.integer, %arg1: !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.integer) {
  // CHECK: return %arg0
  %0 = constant true -> !iree_pydm.bool
  %1 = select %0, %arg0, %arg1 : !iree_pydm.integer
  return %1 : !iree_pydm.integer
}
