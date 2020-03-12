// RUN: check-opt %s -iree-convert-flow-to-hal -iree-vm-conversion -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @expect_true
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @expect_true_i32(%arg : i32) {
  // CHECK: vm.call @check.expect_true([[ARG]]
  check.expect_true(%arg) : i32
  return
}

// -----

// CHECK-LABEL: @expect_false
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @expect_false_i32(%arg : i32) {
  // CHECK: vm.call @check.expect_false([[ARG]]
  check.expect_false(%arg) : i32
  return
}

// -----

// CHECK-LABEL: @expect_all_true
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @expect_all_true(%arg : !hal.buffer_view) {
  // CHECK: vm.call @check.expect_all_true([[ARG]]
  check.expect_all_true(%arg) : !hal.buffer_view
  return
}

// -----

// CHECK-LABEL: @expect_all_true_tensor
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @expect_all_true_tensor(%arg : tensor<2x2xi32>) {
  // CHECK: vm.call @check.expect_all_true(%{{[a-zA-Z0-9]+}}) : (!vm.ref<!hal.buffer_view>) -> ()
  check.expect_all_true(%arg) : tensor<2x2xi32>
  return
}

// -----

// CHECK-LABEL: @expect_eq
// CHECK-SAME: [[LHS:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[RHS:%[a-zA-Z0-9]+]]
func @expect_eq(%lhs : !hal.buffer_view, %rhs : !hal.buffer_view) {
  // CHECK: vm.call @check.expect_eq([[LHS]], [[RHS]]
  check.expect_eq(%lhs, %rhs) : !hal.buffer_view
  return
}

// -----

// CHECK-LABEL: @expect_eq_tensor
// CHECK-SAME: [[LHS:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[RHS:%[a-zA-Z0-9]+]]
func @expect_eq_tensor(%lhs : tensor<2x2xi32>, %rhs : tensor<2x2xi32>) {
  // CHECK: vm.call @check.expect_eq
  // CHECK-SAME: (!vm.ref<!hal.buffer_view>, !vm.ref<!hal.buffer_view>) -> ()
  check.expect_eq(%lhs, %rhs) : tensor<2x2xi32>
  return
}
