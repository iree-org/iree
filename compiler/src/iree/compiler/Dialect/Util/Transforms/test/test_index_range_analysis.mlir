// RUN: iree-opt --split-input-file --iree-util-test-index-range-analysis --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @mul_add
func.func @mul_add(%arg0: index) -> index {
  %0 = arith.constant 10 : index
  %1 = arith.muli %0, %arg0 : index
  // CHECK: arith.muli
  // CHECK-SAME: operand_0 = "lower bound: 10, upper bound: 10"
  %2 = arith.addi %1, %1 : index
  // CHECK: arith.addi
  // CHECK-SAME: operand_0 = "lower bounds: (d0) -> (d0 * 10), upper bounds: (d0) -> (d0 * 10)" 
  return %2 : index
  // CHECK: return
  // CHECK-SAME: operand_0 = "lower bounds: (d0) -> (d0 * 2), upper bounds: (d0) -> (d0 * 2)"
}

// -----

// CHECK-LABEL: @max_min
func.func @max_min(%arg0: index) -> index {
  %2 = affine.max affine_map<(d0) -> (d0, 2)> (%arg0)
  // CHECK: affine.min
  // CHECK-SAME: operand_0 = "lower bound: 2, upper bound: INFINITY"
  %3 = affine.min affine_map<(d0) -> (d0, 10)> (%2)

  // Note that because affine.max doesn't represent a convex integer set, it
  // can't solve for the lower bound here.
  // CHECK: return
  // CHECK-SAME: operand_0 = "lower bound: -INFINITY, upper bound: 10"
  return %3 : index
}

// -----

// CHECK-LABEL: @double_max
func.func @double_max(%arg0: index) -> index {
  %2 = affine.max affine_map<(d0) -> (d0, 10)> (%arg0)
  %3 = affine.max affine_map<(d0) -> (d0, 5)> (%2)
  // CHECK: return
  // CHECK-SAME: operand_0 = "lower bound: 10, upper bound: INFINITY"
  return %3 : index
}

// -----

// CHECK-LABEL: @empty_dim
func.func @empty_dim() -> tensor<?xf32> {
  %c10 = arith.constant 10 : index
  %empty = tensor.empty(%c10) : tensor<?xf32>
  // CHECK: return
  // CHECK-SAME: operand_0_dim_0 = "lower bound: 10, upper bound: 10"
  return %empty : tensor<?xf32>
}

// -----

// CHECK-LABEL: @empty_dim
func.func @empty_dim() -> tensor<?xf32> {
  %c10 = arith.constant 10 : index
  %empty = tensor.empty(%c10) : tensor<?xf32>
  // CHECK: return
  // CHECK-SAME: operand_0_dim_0 = "lower bound: 10, upper bound: 10"
  return %empty : tensor<?xf32>
}

func.func @dim_sum() -> index {
  %c20 = arith.constant 20 : index
  %c0 = arith.constant 0 : index
  %t = func.call @empty_dim() : () -> tensor<?xf32>
  %d = tensor.dim %t, %c0 : tensor<?xf32>
  %sum = arith.addi %d, %c20 : index
  // CHECK: return
  // CHECK-SAME: operand_0 = "lower bound: 30, upper bound: 30"
  return %sum : index
}
