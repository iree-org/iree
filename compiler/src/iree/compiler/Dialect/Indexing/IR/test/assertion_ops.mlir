// RUN: iree-opt --split-input-file %s | FileCheck %s

  // CHECK-LABEL: assert_index
func.func @assert_index(%arg0 : index) -> index {
  // CHECK: indexing.assert.aligned_range %arg0 range(-800, 4) align(11) : index
  %assert = indexing.assert.aligned_range %arg0 range(-800, 4) align(11) : index
  return %assert : index
}

// -----

  // CHECK-LABEL: assert_integer
func.func @assert_integer(%arg0: i32, %arg1: i64) -> (i32, i64) {
  // CHECK: indexing.assert.aligned_range %arg0 range(-1, 2) : i32
  %assert = indexing.assert.aligned_range %arg0 range(-1, 2) : i32
  // CHECK: indexing.assert.aligned_range %arg1 range(-10, 20) align(10) : i64
  %assert1 = indexing.assert.aligned_range %arg1 range(-10, 20) align(10) : i64
  return %assert, %assert1 : i32, i64
}

// -----

  // CHECK-LABEL: assert_optional_bounds
func.func @assert_optional_bounds(%arg0 : index) -> index {
  // CHECK: indexing.assert.aligned_range %{{.*}} range( UNBOUNDED, 4) align(11) : index
  %assert = indexing.assert.aligned_range %arg0 range(UNBOUNDED, 4) align(11) : index
  // CHECK: indexing.assert.aligned_range %{{.*}} range(-800, UNBOUNDED) align(11) : index
  %assert2 = indexing.assert.aligned_range %assert range(-800, UNBOUNDED) align(11) : index
  // CHECK: indexing.assert.aligned_range %{{.*}} range( UNBOUNDED, UNBOUNDED) : index
  %assert3 = indexing.assert.aligned_range %assert2 range(UNBOUNDED, UNBOUNDED) align(1) : index
  return %assert3 : index
}

// -----

  // CHECK-LABEL: assert_dynamic_dim
func.func @assert_dynamic_dim(%arg0: tensor<2x?x3xf32>) -> (tensor<2x?x3xf32>) {
  // CHECK: indexing.assert.dim_range %arg0[1] range(10, 20) align(2) : tensor<2x?x3xf32>
  %assert = indexing.assert.dim_range %arg0[1] range(10, 20) align(2) : tensor<2x?x3xf32>
  return %assert : tensor<2x?x3xf32>
}
