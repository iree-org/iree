// RUN: iree-opt --split-input-file --iree-util-test-float-range-analysis --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @scalar_const_trunc
func.func @scalar_const_trunc() -> f32 {
  %0 = arith.constant 5.0 : f32
  // CHECK: fp-range: [5.000000, 5.000000, TRUNC]
  %1 = "iree_unregistered.test_fprange"(%0) : (f32) -> f32
  return %1 : f32
}

// -----
// CHECK-LABEL: @scalar_const_non_trunc
func.func @scalar_const_non_trunc() -> f32 {
  %0 = arith.constant 5.2 : f32
  // CHECK: fp-range: [5.200000, 5.200000, !trunc]
  %1 = "iree_unregistered.test_fprange"(%0) : (f32) -> f32
  return %1 : f32
}

// -----
// CHECK-LABEL: @scalar_non_float
func.func @scalar_non_float() -> i32 {
  %0 = arith.constant 5 : i32
  // NOTE: The least-constrained value is returned for a non-fp type. It
  // is up to the user to ensure that we are requesting stats for fp types
  // and this represents the soft-failure mode if asking about an illegal type.
  // CHECK: fp-range: [-inf, inf, !trunc]
  %1 = "iree_unregistered.test_fprange"(%0) : (i32) -> i32
  return %1 : i32
}

// -----
// CHECK-LABEL: @tensor_const_trunc
func.func @tensor_const_trunc() -> tensor<2xf32> {
  %0 = arith.constant dense<[-2.0, 2.0]> : tensor<2xf32>
  // CHECK: fp-range: [-2.000000, 2.000000, TRUNC]
  %1 = "iree_unregistered.test_fprange"(%0) : (tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

// -----
// CHECK-LABEL: @tensor_const_non_trunc
func.func @tensor_const_non_trunc() -> tensor<2xf32> {
  %0 = arith.constant dense<[-1.2, 2.0]> : tensor<2xf32>
  // CHECK: fp-range: [-1.200000, 2.000000, !trunc]
  %1 = "iree_unregistered.test_fprange"(%0) : (tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

// -----
// CHECK-LABEL: @min_max_no_trunc
func.func @min_max_no_trunc(%arg0 : f32) -> f32 {
  %0 = arith.constant -5.0 : f32
  %1 = arith.constant 5.0 : f32
  %2 = arith.minimumf %arg0, %1 : f32
  %3 = arith.maximumf %2, %0 : f32
  // CHECK: fp-range: [-5.000000, 5.000000, !trunc]
  %result = "iree_unregistered.test_fprange"(%3) : (f32) -> f32
  return %result : f32
}

// -----
// CHECK-LABEL: @min_max_floor
func.func @min_max_floor(%arg0 : f32) -> f32 {
  %0 = arith.constant -5.0 : f32
  %1 = arith.constant 5.0 : f32
  %2 = arith.minimumf %arg0, %1 : f32
  %3 = arith.maximumf %2, %0 : f32
  %4 = math.floor %3 : f32
  // CHECK: fp-range: [-5.000000, 5.000000, TRUNC]
  %result = "iree_unregistered.test_fprange"(%4) : (f32) -> f32
  return %result : f32
}

// -----
// CHECK-LABEL: @min_max_floor_adj_range
func.func @min_max_floor_adj_range(%arg0 : f32) -> f32 {
  %0 = arith.constant -5.2 : f32
  %1 = arith.constant 5.2 : f32
  %2 = arith.minimumf %arg0, %1 : f32
  %3 = arith.maximumf %2, %0 : f32
  %4 = math.floor %3 : f32
  // CHECK: fp-range: [-6.000000, 5.000000, TRUNC]
  %result = "iree_unregistered.test_fprange"(%4) : (f32) -> f32
  return %result : f32
}

// -----
// CHECK-LABEL: @floor_min_max
func.func @floor_min_max(%arg0 : f32) -> f32 {
  %0 = arith.constant -5.0 : f32
  %1 = arith.constant 5.0 : f32
  %2 = math.floor %arg0 : f32
  %3 = arith.maximumf %2, %0 : f32
  %4 = arith.minimumf %3, %1 : f32
  // CHECK: fp-range: [-5.000000, 5.000000, TRUNC]
  %result = "iree_unregistered.test_fprange"(%4) : (f32) -> f32
  return %result : f32
}
