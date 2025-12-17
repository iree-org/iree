// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-expand-dimensions))" | FileCheck %s

func.func @expand_matvec(%a: tensor<4x16384xf16>, %b: tensor<1x16384xf16>) -> tensor<4x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4x1xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x1xf32>) -> tensor<4x1xf32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<4x16384xf16>, tensor<1x16384xf16>)
    outs(%fill : tensor<4x1xf32>)
    attrs = {
      lowering_config = #iree_gpu.lowering_config<{
      expand_dims = #iree_gpu.expand_dims<[[0], [1], [2, 3]], output_shape = [?, ?, ?, 8]>,
      lane_basis = [[1, 1, 64, 1], [0, 1, 2, 3]],
      partial_reduction = [0, 0, 64, 0],
      subgroup_basis = [[1, 1, 1, 1], [0, 1, 2, 3]],
      thread = [0, 0, 1, 8],
      workgroup = [4, 1, 0, 0]}>} {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %0 = arith.extf %in : f16 to f32
    %1 = arith.extf %in_0 : f16 to f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.addf %out, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}

// CHECK-LABEL: func.func @expand_matvec
// CHECK: %[[A_EXPAND:.*]] = tensor.expand_shape %{{.*}} {{\[}}[0], [1, 2]] output_shape [4, 2048, 8] : tensor<4x16384xf16> into tensor<4x2048x8xf16>
// CHECK: %[[B_EXPAND:.*]] = tensor.expand_shape %{{.*}} {{\[}}[0], [1, 2]] output_shape [1, 2048, 8] : tensor<1x16384xf16> into tensor<1x2048x8xf16>
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[A_EXPAND]], %[[B_EXPAND]] : tensor<4x2048x8xf16>, tensor<1x2048x8xf16>)

// -----

func.func @expand_multiple_dims(%a: tensor<4x16384xf16>, %b: tensor<4x16384xf16>) -> tensor<4x16384xf16> {
  %empty = tensor.empty() : tensor<4x16384xf16>
  %result = linalg.add {
    lowering_config = #iree_gpu.lowering_config<{
      expand_dims = #iree_gpu.expand_dims<[[0], [1, 2, 3]], output_shape = [?, ?, 2, 4]>
    }>}
    ins(%a, %b : tensor<4x16384xf16>, tensor<4x16384xf16>) outs(%empty : tensor<4x16384xf16>) -> tensor<4x16384xf16>
  return %result : tensor<4x16384xf16>
}

// CHECK-LABEL: func.func @expand_multiple_dims
// CHECK: %[[A_EXPAND:.*]] = tensor.expand_shape %{{.*}} {{\[}}[0], [1, 2, 3]] output_shape [4, 2048, 2, 4] : tensor<4x16384xf16> into tensor<4x2048x2x4xf16>
// CHECK: %[[B_EXPAND:.*]] = tensor.expand_shape %{{.*}} {{\[}}[0], [1, 2, 3]] output_shape [4, 2048, 2, 4] : tensor<4x16384xf16> into tensor<4x2048x2x4xf16>
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[A_EXPAND]], %[[B_EXPAND]] : tensor<4x2048x2x4xf16>, tensor<4x2048x2x4xf16>)

// -----

// Verify that dynamic dimensions are gracefully handled (no expansion occurs).
func.func @no_expand_dynamic_dims(%a: tensor<4x?xf16>, %b: tensor<4x?xf16>) -> tensor<4x128xf16> {
  %empty = tensor.empty() : tensor<4x128xf16>
  %result = linalg.add {
    lowering_config = #iree_gpu.lowering_config<{
      expand_dims = #iree_gpu.expand_dims<[[0], [1, 2]], output_shape = [?, ?, 8]>
    }>}
    ins(%a, %b : tensor<4x?xf16>, tensor<4x?xf16>) outs(%empty : tensor<4x128xf16>) -> tensor<4x128xf16>
  return %result : tensor<4x128xf16>
}

// CHECK-LABEL: func.func @no_expand_dynamic_dim
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.add
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<4x?xf16>, tensor<4x?xf16>)

// -----

// Verify that non-divisible dimensions are gracefully handled (no expansion occurs).
func.func @no_expand_not_divisible(%a: tensor<4x127xf16>, %b: tensor<4x127xf16>) -> tensor<4x127xf16> {
  %empty = tensor.empty() : tensor<4x127xf16>
  %result = linalg.add {
    lowering_config = #iree_gpu.lowering_config<{
      expand_dims = #iree_gpu.expand_dims<[[0], [1, 2]], output_shape = [?, ?, 8]>
    }>}
    ins(%a, %b : tensor<4x127xf16>, tensor<4x127xf16>) outs(%empty : tensor<4x127xf16>) -> tensor<4x127xf16>
  return %result : tensor<4x127xf16>
}

// CHECK-LABEL: func.func @no_expand_not_divisible
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.add
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<4x127xf16>, tensor<4x127xf16>)
