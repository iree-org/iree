// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-stablehlo-preprocessing-gather-to-torch-index-select))" %s \
// RUN:   | FileCheck %s
// RUN: iree-opt --iree-stablehlo-input-transformation-pipeline %s \
// RUN:   | FileCheck %s --check-prefix=PIPELINE --implicit-check-not=stablehlo.

// CHECK-LABEL: @gather_dynamic_result
// CHECK: "stablehlo.gather"
// PIPELINE-LABEL: @gather_dynamic_result
// PIPELINE: tensor.dim %arg1
// PIPELINE: tensor.empty(%{{.+}}) : tensor<?x4xf32>
// PIPELINE: linalg.generic
// PIPELINE: arith.maxsi
// PIPELINE: arith.minsi
// PIPELINE: return %{{.+}} : tensor<?x4xf32>
func.func @gather_dynamic_result(%arg0: tensor<?x4xf32>, %arg1: tensor<?x1xi64>) -> tensor<?x4xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>,
    indices_are_sorted = false
  } : (tensor<?x4xf32>, tensor<?x1xi64>) -> tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

// CHECK-LABEL: @gather_to_index_select
func.func @gather_to_index_select(%arg0 : tensor<5x4xf32>, %arg1 : tensor<1x3x1xi32>) -> tensor<1x3x4xf32> {
  // CHECK-DAG: [[ZERO:%.+]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-DAG: [[MAX:%.+]] = stablehlo.constant dense<4> : tensor<i64>
  // CHECK: [[WIDE:%.+]] = stablehlo.convert %arg1 : (tensor<1x3x1xi32>) -> tensor<1x3x1xi64>
  // CHECK: [[CLAMP:%.+]] = stablehlo.clamp [[ZERO]], [[WIDE]], [[MAX]]
  // CHECK: [[TIS:%.+]] = "stablehlo.torch_index_select"(%arg0, [[CLAMP]])
  // CHECK-SAME:   batch_dims = 0 : i64,
  // CHECK-SAME:   dim = 0 : i64
  // CHECK-SAME: : (tensor<5x4xf32>, tensor<1x3x1xi64>) -> tensor<1x3x1x4xf32>
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

// CHECK-LABEL: @gather_dynamic_operand
// CHECK: "stablehlo.gather"
// PIPELINE-LABEL: @gather_dynamic_operand
// PIPELINE: tensor.dim %arg0
// PIPELINE: arith.maxsi {{.*}} : i64
// PIPELINE: arith.minsi {{.*}} : i64
// PIPELINE: arith.index_cast {{.*}} : i64 to index
func.func @gather_dynamic_operand(%arg0: tensor<?x4xf32>, %arg1: tensor<3x1xi64>) -> tensor<3x4xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], index_vector_dim = 1, offset_dims = [1], start_index_map = [0]>,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<?x4xf32>, tensor<3x1xi64>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// A static bound makes source-width clamping safe for index_select on both
// 32-bit and 64-bit targets.
// CHECK-LABEL: @gather_wide_indices
// CHECK: %[[CLAMP:.+]] = stablehlo.clamp
// CHECK: "stablehlo.torch_index_select"(%arg0, %[[CLAMP]])
// PIPELINE-LABEL: @gather_wide_indices
// PIPELINE: arith.maxsi
// PIPELINE: arith.minsi
func.func @gather_wide_indices(%arg0: tensor<5x4xf32>, %arg1: tensor<2x1xi64>) -> tensor<2x4xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], index_vector_dim = 1, offset_dims = [1], start_index_map = [0]>,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<5x4xf32>, tensor<2x1xi64>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: @gather_unsigned_indices
// CHECK: %[[CLAMP:.+]] = stablehlo.clamp
// CHECK: "stablehlo.torch_index_select"(%arg0, %[[CLAMP]])
// PIPELINE-LABEL: @gather_unsigned_indices
// PIPELINE: arith.minui {{.*}} : i64
func.func @gather_unsigned_indices(%arg0: tensor<5x4xf32>, %arg1: tensor<2x1xui64>) -> tensor<2x4xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], index_vector_dim = 1, offset_dims = [1], start_index_map = [0]>,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<5x4xf32>, tensor<2x1xui64>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: @gather_large_operand
// CHECK: "stablehlo.gather"
// PIPELINE-LABEL: @gather_large_operand
// PIPELINE: arith.minsi {{.*}} : i64
func.func @gather_large_operand(%arg0: tensor<2147483649x4xf32>, %arg1: tensor<2x1xi64>) -> tensor<2x4xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], index_vector_dim = 1, offset_dims = [1], start_index_map = [0]>,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<2147483649x4xf32>, tensor<2x1xi64>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
