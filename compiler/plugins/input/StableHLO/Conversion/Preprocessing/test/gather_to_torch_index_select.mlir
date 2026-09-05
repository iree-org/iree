// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-stablehlo-preprocessing-gather-to-torch-index-select))" %s \
// RUN:   | FileCheck %s

// CHECK-LABEL: @gather_to_index_select
func.func @gather_to_index_select(%arg0 : tensor<5x4xf32>, %arg1 : tensor<1x3x1xi32>) -> tensor<1x3x4xf32> {
  // The spec clamps start indices into [0, operand_dim - slice_size], here
  // [0, 4]. torch_index_select does no bounds checking of its own.
  // CHECK-DAG: [[HI:%.+]] = stablehlo.constant dense<4> : tensor<i32>
  // CHECK-DAG: [[LO:%.+]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: [[IDX:%.+]] = stablehlo.clamp [[LO]], %arg1, [[HI]] :
  // CHECK-SAME: (tensor<i32>, tensor<1x3x1xi32>, tensor<i32>) -> tensor<1x3x1xi32>
  // CHECK: [[TIS:%.+]] = "stablehlo.torch_index_select"(%arg0, [[IDX]])
  // CHECK-SAME:   batch_dims = 0 : i64,
  // CHECK-SAME:   dim = 0 : i64
  // CHECK-SAME: : (tensor<5x4xf32>, tensor<1x3x1xi32>) -> tensor<1x3x1x4xf32>
  // CHECK: [[RES:%.+]] = stablehlo.reshape [[TIS]]
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<5x4xf32>, tensor<1x3x1xi32>) -> tensor<1x3x4xf32>

  // CHECK: return [[RES]]
  func.return %0 : tensor<1x3x4xf32>
}

// The bound takes the element type of the indices.
// CHECK-LABEL: @gather_to_index_select_i64
func.func @gather_to_index_select_i64(%arg0 : tensor<5x4xf32>, %arg1 : tensor<1x3x1xi64>) -> tensor<1x3x4xf32> {
  // CHECK-DAG: [[HI:%.+]] = stablehlo.constant dense<4> : tensor<i64>
  // CHECK-DAG: [[LO:%.+]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK: [[IDX:%.+]] = stablehlo.clamp [[LO]], %arg1, [[HI]]
  // CHECK: "stablehlo.torch_index_select"(%arg0, [[IDX]])
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<5x4xf32>, tensor<1x3x1xi64>) -> tensor<1x3x4xf32>
  func.return %0 : tensor<1x3x4xf32>
}

// An i8 index cannot name element 199, so the bound saturates at the largest
// i8 rather than wrapping around into a negative one.
// CHECK-LABEL: @gather_to_index_select_narrow_index
func.func @gather_to_index_select_narrow_index(%arg0 : tensor<200x4xf32>, %arg1 : tensor<1x3x1xi8>) -> tensor<1x3x4xf32> {
  // CHECK-DAG: [[HI:%.+]] = stablehlo.constant dense<127> : tensor<i8>
  // CHECK-DAG: [[LO:%.+]] = stablehlo.constant dense<0> : tensor<i8>
  // CHECK: [[IDX:%.+]] = stablehlo.clamp [[LO]], %arg1, [[HI]]
  // CHECK: "stablehlo.torch_index_select"(%arg0, [[IDX]])
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<200x4xf32>, tensor<1x3x1xi8>) -> tensor<1x3x4xf32>
  func.return %0 : tensor<1x3x4xf32>
}

// A dynamic gathered dimension takes its bound from the operand at runtime.
// CHECK-LABEL: @gather_to_index_select_dynamic
func.func @gather_to_index_select_dynamic(%arg0 : tensor<?x4xf32>, %arg1 : tensor<1x3x1xi32>) -> tensor<1x3x4xf32> {
  // CHECK-DAG: [[ONE:%.+]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK-DAG: [[LO:%.+]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: [[SIZE:%.+]] = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x4xf32>) -> tensor<i32>
  // CHECK: [[HI:%.+]] = stablehlo.subtract [[SIZE]], [[ONE]] : tensor<i32>
  // CHECK: [[IDX:%.+]] = stablehlo.clamp [[LO]], %arg1, [[HI]]
  // CHECK: "stablehlo.torch_index_select"(%arg0, [[IDX]])
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<?x4xf32>, tensor<1x3x1xi32>) -> tensor<1x3x4xf32>
  func.return %0 : tensor<1x3x4xf32>
}

// An index type narrower than the i32 get_dimension_size returns cannot hold
// the bound, so leave the gather for the general lowering.
// CHECK-LABEL: @gather_no_lowering_dynamic_narrow_index
func.func @gather_no_lowering_dynamic_narrow_index(%arg0 : tensor<?x4xf32>, %arg1 : tensor<1x3x1xi8>) -> tensor<1x3x4xf32> {
  // CHECK: "stablehlo.gather"
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<?x4xf32>, tensor<1x3x1xi8>) -> tensor<1x3x4xf32>
  func.return %0 : tensor<1x3x4xf32>
}

// CHECK-LABEL: @gather_no_lowering_subslice
func.func @gather_no_lowering_subslice(%arg0 : tensor<5x4xf32>, %arg1 : tensor<1x3x1xi32>) -> tensor<1x3x3xf32> {
  // CHECK: "stablehlo.gather"
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 3>
  } : (tensor<5x4xf32>, tensor<1x3x1xi32>) -> tensor<1x3x3xf32>
  func.return %0 : tensor<1x3x3xf32>
}

// CHECK-LABEL: @gather_no_lowering_multidim
func.func @gather_no_lowering_multidim(%arg0 : tensor<5x4xf32>, %arg1 : tensor<1x3x2xi32>) -> tensor<1x3x4xf32> {
  // CHECK: "stablehlo.gather"
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<5x4xf32>, tensor<1x3x2xi32>) -> tensor<1x3x4xf32>
  func.return %0 : tensor<1x3x4xf32>
}
