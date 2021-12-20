// RUN: iree-dialects-opt -split-input-file --allow-unregistered-dialect -canonicalize %s | FileCheck  --dump-input-filter=all %s

// CHECK-LABEL: @dynamic_binary_promote_same_type
iree_pydm.func @dynamic_binary_promote_same_type(%arg0 : !iree_pydm.bool, %arg1 : !iree_pydm.bool) -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // Note: it is important that types are not modified as part of canonicalization,
  // since the legality of that requires more analysis. Therefore, this must
  // produce unrefined objects, like the original.
  // CHECK: %[[LEFT:.*]] = box %arg0 : !iree_pydm.bool -> <!iree_pydm.bool>
  // CHECK: %[[LEFT_CASTED:.*]] = static_info_cast %[[LEFT]] : !iree_pydm.object<!iree_pydm.bool> -> !iree_pydm.object
  // CHECK: %[[RIGHT:.*]] = box %arg1 : !iree_pydm.bool -> <!iree_pydm.bool>
  // CHECK: %[[RIGHT_CASTED:.*]] = static_info_cast %[[RIGHT]] : !iree_pydm.object<!iree_pydm.bool> -> !iree_pydm.object
  // CHECK: "custom.donotoptimize"(%[[LEFT_CASTED]], %[[RIGHT_CASTED]])
  %0, %1 = dynamic_binary_promote %arg0, %arg1 : !iree_pydm.bool, !iree_pydm.bool
  "custom.donotoptimize"(%0, %1) : (!iree_pydm.object, !iree_pydm.object) -> ()
  %none = none
  return %none : !iree_pydm.none
}

// -----
// CHECK-LABEL: @dynamic_binary_promote_promote_left
iree_pydm.func @dynamic_binary_promote_promote_left(%arg0 : !iree_pydm.bool, %arg1 : !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK: %[[LEFT:.*]] = promote_numeric %arg0 : !iree_pydm.bool -> !iree_pydm.integer
  // CHECK: %[[LEFT_BOXED:.*]] = box %[[LEFT]] : !iree_pydm.integer -> <!iree_pydm.integer>
  // CHECK: %[[LEFT_CASTED:.*]] = static_info_cast %[[LEFT_BOXED]] : !iree_pydm.object<!iree_pydm.integer> -> !iree_pydm.object
  // CHECK: %[[RIGHT_BOXED:.*]] = box %arg1 : !iree_pydm.integer -> <!iree_pydm.integer>
  // CHECK: %[[RIGHT_CASTED:.*]] = static_info_cast %[[RIGHT_BOXED]] : !iree_pydm.object<!iree_pydm.integer> -> !iree_pydm.object
  // CHECK: "custom.donotoptimize"(%[[LEFT_CASTED]], %[[RIGHT_CASTED]])
  %0, %1 = dynamic_binary_promote %arg0, %arg1 : !iree_pydm.bool, !iree_pydm.integer
  "custom.donotoptimize"(%0, %1) : (!iree_pydm.object, !iree_pydm.object) -> ()
  %none = none
  return %none : !iree_pydm.none
}

// -----
// CHECK-LABEL: @dynamic_binary_promote_promote_right
iree_pydm.func @dynamic_binary_promote_promote_right(%arg0 : !iree_pydm.real, %arg1 : !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK: %[[RIGHT:.*]] = promote_numeric %arg1 : !iree_pydm.integer -> !iree_pydm.real
  // CHECK: %[[LEFT_BOXED:.*]] = box %arg0 : !iree_pydm.real -> <!iree_pydm.real>
  // CHECK: %[[LEFT_CASTED:.*]] = static_info_cast %[[LEFT_BOXED]] : !iree_pydm.object<!iree_pydm.real> -> !iree_pydm.object
  // CHECK: %[[RIGHT_BOXED:.*]] = box %[[RIGHT]] : !iree_pydm.real -> <!iree_pydm.real>
  // CHECK: %[[RIGHT_CASTED:.*]] = static_info_cast %[[RIGHT_BOXED]] : !iree_pydm.object<!iree_pydm.real> -> !iree_pydm.object
  // CHECK: "custom.donotoptimize"(%[[LEFT_CASTED]], %[[RIGHT_CASTED]])
  %0, %1 = dynamic_binary_promote %arg0, %arg1 : !iree_pydm.real, !iree_pydm.integer
  "custom.donotoptimize"(%0, %1) : (!iree_pydm.object, !iree_pydm.object) -> ()
  %none = none
  return %none : !iree_pydm.none
}

// -----
// CHECK-LABEL: @dynamic_binary_promote_promote_non_numeric
// If either operand is a concrete, non numeric type, then the op is elided
// (which to satisfy typing, involves boxing and static casting).
iree_pydm.func @dynamic_binary_promote_promote_non_numeric(%arg0 : !iree_pydm.real, %arg1 : !iree_pydm.none) -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK-NOT: dynamic_binary_promote
  // CHECK-NOT: promote_numeric
  %0, %1 = dynamic_binary_promote %arg0, %arg1 : !iree_pydm.real, !iree_pydm.none
  %2, %3 = dynamic_binary_promote %arg1, %arg0 : !iree_pydm.none, !iree_pydm.real
  "custom.donotoptimize"(%0, %1, %2, %3) : (!iree_pydm.object, !iree_pydm.object, !iree_pydm.object, !iree_pydm.object) -> ()
  %none = none
  return %none : !iree_pydm.none
}

// -----
// CHECK-LABEL: @elide_promote_numeric_identity
iree_pydm.func @elide_promote_numeric_identity(%arg0: !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK: "custom.donotoptimize"(%arg0)
  %0 = promote_numeric %arg0 : !iree_pydm.integer -> !iree_pydm.integer
  "custom.donotoptimize"(%0) : (!iree_pydm.integer) -> ()
  %none = none
  return %none : !iree_pydm.none
}

// -----
// CHECK-LABEL: @fold_promote_numeric_true_to_integer
iree_pydm.func @fold_promote_numeric_true_to_integer() -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK: %[[CST:.*]] = constant 1 : i64 -> !iree_pydm.integer
  // CHECK: "custom.donotoptimize"(%[[CST]])
  %0 = constant true -> !iree_pydm.bool
  %1 = promote_numeric %0 : !iree_pydm.bool -> !iree_pydm.integer
  "custom.donotoptimize"(%1) : (!iree_pydm.integer) -> ()
  %none = none
  return %none : !iree_pydm.none
}

// -----
// CHECK-LABEL: @fold_promote_numeric_false_to_integer
iree_pydm.func @fold_promote_numeric_false_to_integer() -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK: %[[CST:.*]] = constant 0 : i64 -> !iree_pydm.integer
  // CHECK: "custom.donotoptimize"(%[[CST]])
  %0 = constant false -> !iree_pydm.bool
  %1 = promote_numeric %0 : !iree_pydm.bool -> !iree_pydm.integer
  "custom.donotoptimize"(%1) : (!iree_pydm.integer) -> ()
  %none = none
  return %none : !iree_pydm.none
}

// -----
// CHECK-LABEL: @fold_promote_numeric_true_to_real
iree_pydm.func @fold_promote_numeric_true_to_real() -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK: %[[CST:.*]] = constant 1.000000e+00 : f64 -> !iree_pydm.real
  // CHECK: "custom.donotoptimize"(%[[CST]])
  %0 = constant true -> !iree_pydm.bool
  %1 = promote_numeric %0 : !iree_pydm.bool -> !iree_pydm.real
  "custom.donotoptimize"(%1) : (!iree_pydm.real) -> ()
  %none = none
  return %none : !iree_pydm.none
}

// -----
// CHECK-LABEL: @fold_promote_numeric_false_to_real
iree_pydm.func @fold_promote_numeric_false_to_real() -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK: %[[CST:.*]] = constant 0.000000e+00 : f64 -> !iree_pydm.real
  // CHECK: "custom.donotoptimize"(%[[CST]])
  %0 = constant false -> !iree_pydm.bool
  %1 = promote_numeric %0 : !iree_pydm.bool -> !iree_pydm.real
  "custom.donotoptimize"(%1) : (!iree_pydm.real) -> ()
  %none = none
  return %none : !iree_pydm.none
}

// -----
// CHECK-LABEL: @fold_promote_numeric_integet_to_real
iree_pydm.func @fold_promote_numeric_integet_to_real() -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK: %[[CST:.*]] = constant 2.000000e+00 : f64 -> !iree_pydm.real
  // CHECK: "custom.donotoptimize"(%[[CST]])
  %0 = constant 2 : i64 -> !iree_pydm.integer
  %1 = promote_numeric %0 : !iree_pydm.integer -> !iree_pydm.real
  "custom.donotoptimize"(%1) : (!iree_pydm.real) -> ()
  %none = none
  return %none : !iree_pydm.none
}

// -----
// CHECK-LABEL: @apply_binary_mul_to_sequence_clone
iree_pydm.func @apply_binary_mul_to_sequence_clone(%arg0 : !iree_pydm.list, %arg1 : !iree_pydm.integer<32>) -> (!iree_pydm.exception_result, !iree_pydm.none) {
  // CHECK: sequence_clone %arg0 * %arg1 : !iree_pydm.list, !iree_pydm.integer<32> -> !iree_pydm.object
  %0 = apply_binary "mul", %arg0, %arg1 : !iree_pydm.list, !iree_pydm.integer<32> -> !iree_pydm.object
  // Regardless of order to apply_binary, matches.
  // CHECK: sequence_clone %arg0 * %arg1 : !iree_pydm.list, !iree_pydm.integer<32> -> !iree_pydm.object
  %1 = apply_binary "mul", %arg1, %arg0 : !iree_pydm.integer<32>, !iree_pydm.list -> !iree_pydm.object
  // Only mul is matched.
  // CHECK: apply_binary "add"
  %2 = apply_binary "add", %arg1, %arg0 : !iree_pydm.integer<32>, !iree_pydm.list -> !iree_pydm.object
  "custom.donotoptimize"(%0, %1, %2) : (!iree_pydm.object, !iree_pydm.object, !iree_pydm.object) -> ()
  %none = none
  return %none : !iree_pydm.none
}
