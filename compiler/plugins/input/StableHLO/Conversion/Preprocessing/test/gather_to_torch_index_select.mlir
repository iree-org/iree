// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-stablehlo-preprocessing-gather-to-torch-index-select))" %s \
// RUN:   | FileCheck %s
// RUN: iree-opt --iree-stablehlo-input-transformation-pipeline %s | FileCheck %s --check-prefix=LOWER

// CHECK-LABEL: @gather_to_index_select
func.func @gather_to_index_select(%arg0 : tensor<5x4xf32>, %arg1 : tensor<1x3x1xi32>) -> tensor<1x3x4xf32> {
  // CHECK: [[TIS:%.+]] = "stablehlo.torch_index_select"(%arg0, %arg1)
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

// The static reshape used by the index-select shortcut cannot represent this result.
// CHECK-LABEL: @dynamic_indices
// CHECK-NOT: stablehlo.torch_index_select
// CHECK-NOT: stablehlo.reshape
// CHECK: stablehlo.gather
// LOWER-LABEL: @dynamic_indices
// LOWER: tensor.dim
// LOWER: linalg.generic
// LOWER: arith.maxsi
// LOWER: arith.minsi
// LOWER: tensor.extract
func.func @dynamic_indices(%a: tensor<?x4xf32>, %i: tensor<?x1xi64>) -> tensor<?x4xf32> {
  %r = "stablehlo.gather"(%a, %i) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>, indices_are_sorted = false
  } : (tensor<?x4xf32>, tensor<?x1xi64>) -> tensor<?x4xf32>
  return %r : tensor<?x4xf32>
}

// The static reshape used by the index-select shortcut cannot represent this result.
// CHECK-LABEL: @dynamic_indices_higher_rank
// CHECK-NOT: stablehlo.torch_index_select
// CHECK-NOT: stablehlo.reshape
// CHECK: stablehlo.gather
// LOWER-LABEL: @dynamic_indices_higher_rank
// LOWER: tensor.dim
// LOWER: linalg.generic
// LOWER: arith.maxsi
// LOWER: arith.minsi
// LOWER: tensor.extract
func.func @dynamic_indices_higher_rank(%a: tensor<5x4xf32>, %i: tensor<2x?x1xi32>) -> tensor<2x?x4xf32> {
  %r = "stablehlo.gather"(%a, %i) {
    dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 4>, indices_are_sorted = false
  } : (tensor<5x4xf32>, tensor<2x?x1xi32>) -> tensor<2x?x4xf32>
  return %r : tensor<2x?x4xf32>
}
