// RUN: iree-opt --verify-diagnostics --canonicalize --split-input-file %s | FileCheck %s

// CHECK-LABEL: @no_fold_constant
util.func public @no_fold_constant() -> (i32) {
  // CHECK: constant 1 : i32
  %0 = arith.constant 1 : i32
  // CHECK: util.optimization_barrier
  %1 = "util.optimization_barrier"(%0) : (i32) -> i32
  util.return %1 : i32
}

// -----

// CHECK-LABEL: @no_fold_add
util.func public @no_fold_add() -> (i32) {
  // CHECK-NEXT: %[[C1:.+]] = vm.const.i32 1
  %c1 = vm.const.i32 1
  // CHECK-NEXT: %[[R1:.+]] = util.optimization_barrier %[[C1]]
  %0 = util.optimization_barrier %c1 : i32
  // CHECK-NEXT: %[[R2:.+]] = vm.add.i32 %[[R1]], %[[R1]]
  %1 = vm.add.i32 %0, %0 : i32
  // CHECK-NEXT: util.return %[[R2]]
  util.return %1 : i32
}

// -----

// Exists to check that the above succeeds when there's no barrier.
// CHECK-LABEL: @fold_add
util.func public @fold_add() -> (i32) {
  // CHECK-NEXT: %[[C2:.+]] = vm.const.i32 2
  // CHECK-NEXT: util.return %[[C2]]
  %c1 = vm.const.i32 1
  %0 = vm.add.i32 %c1, %c1 : i32
  util.return %0 : i32
}

// -----

util.func public @result_operand_count_mismatch(%arg0 : tensor<i32>, %arg1 : tensor<i32>) {
  // expected-error@+1 {{must have same number of operands and results}}
  %1 = "util.optimization_barrier"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  util.return
}

// -----

util.func public @result_operand_type_mismatch(%arg0 : tensor<i32>, %arg1 : tensor<i32>) {
  // expected-error@+1 {{must have same operand and result types, but they differ at index 1}}
  %1:2 = "util.optimization_barrier"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, memref<i32>)
  util.return
}

// -----

// CHECK-LABEL: @canonicalize_unfoldable_constant
util.func public @canonicalize_unfoldable_constant() -> i32 {
  // CHECK-NEXT: %[[C:.+]] = arith.constant 42 : i32
  // CHECK-NEXT: %[[R:.+]] = util.optimization_barrier %[[C]] : i32
  %c42 = util.unfoldable_constant 42 : i32
  // CHECK-NEXT: util.return %[[R]]
  util.return %c42 : i32
}
