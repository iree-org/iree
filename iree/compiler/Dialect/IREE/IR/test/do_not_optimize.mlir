// RUN: iree-opt -verify-diagnostics -canonicalize -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: @no_fold_constant
func @no_fold_constant() -> (i32) {
  // CHECK: constant 1 : i32
  %0 = constant 1 : i32
  // CHECK: iree.do_not_optimize
  %1 = "iree.do_not_optimize"(%0) : (i32) -> i32
  return %1 : i32
}

// -----

// CHECK-LABEL: @no_fold_add
func @no_fold_add() -> (i32) {
  // CHECK-NEXT: %[[C1:.+]] = vm.const.i32 1 : i32
  %c1 = vm.const.i32 1 : i32
  // CHECK-NEXT: %[[R1:.+]] = iree.do_not_optimize(%[[C1]])
  %0 = iree.do_not_optimize(%c1) : i32
  // CHECK-NEXT: %[[R2:.+]] = vm.add.i32 %[[R1]], %[[R1]]
  %1 = vm.add.i32 %0, %0 : i32
  // CHECK-NEXT: return %[[R2]]
  return %1 : i32
}

// -----

// Exists to check that the above succeeds because of do_not_optimize
// CHECK-LABEL: @fold_add
func @fold_add() -> (i32) {
  // CHECK-NEXT: %[[C2:.+]] = vm.const.i32 2
  // CHECK-NEXT: return %[[C2]]
  %c1 = vm.const.i32 1 : i32
  %0 = vm.add.i32 %c1, %c1 : i32
  return %0 : i32
}

// -----

func @result_operand_count_mismatch(%arg0 : tensor<i32>, %arg1 : tensor<i32>) {
  // expected-error@+1 {{must have same number of operands and results}}
  %1 = "iree.do_not_optimize"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return
}

// -----

func @result_operand_type_mismatch(%arg0 : tensor<i32>, %arg1 : tensor<i32>) {
  // expected-error@+1 {{must have same operand and result types, but they differ at index 1}}
  %1:2 = "iree.do_not_optimize"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, memref<i32>)
  return
}

// -----

// CHECK-LABEL: @canonicalize_unfoldable_constant
func @canonicalize_unfoldable_constant() -> i32 {
  // CHECK-NEXT: %[[C:.+]] = constant 42 : i32
  // CHECK-NEXT: %[[R:.+]] = iree.do_not_optimize(%[[C]]) : i32
  %c42 = iree.unfoldable_constant 42 : i32
  // CHECK-NEXT: return %[[R]]
  return %c42 : i32
}
