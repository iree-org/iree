// RUN: iree-opt -split-input-file -iree-hlo-to-linalg %s | IreeFileCheck %s

// CHECK: [[MAP0:#.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 4 + d2 * 2 + d3, d4)>
// CHECK: [[MAP1:#.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

// CHECK: func @reshape_expand_dim
func @reshape_expand_dim(%arg0 : tensor<1x16x2xf32>) -> tensor<1x4x2x2x2xf32>  {
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<1x16x2xf32>) -> tensor<1x4x2x2x2xf32>
  return %0 : tensor<1x4x2x2x2xf32>
}
// CHECK: linalg.generic {
// CHECK-SAME: args_in = 1
// CHECK-SAME: args_out = 1
// CHECK-SAME: indexing_maps
// CHECK-SAME: [[MAP0]], [[MAP1]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
// CHECK-NEXT: ^{{.+}}([[ARG:%.*]]: f32)
// CHECK-NEXT: linalg.yield [[ARG]] : f32

// -----

// CHECK: [[MAP2:#.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK: [[MAP3:#.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @reshape_expand_single_dim
func @reshape_expand_single_dim(%arg0 : tensor<8xf32>) -> tensor<4x2xf32>  {
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<8xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}
// CHECK: linalg.generic {
// CHECK-SAME: args_in = 1
// CHECK-SAME: args_out = 1
// CHECK-SAME: indexing_maps
// CHECK-SAME: [[MAP2]], [[MAP3]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-NEXT: ^{{.+}}([[ARG:%.*]]: f32)
// CHECK-NEXT: linalg.yield [[ARG]] : f32

// -----

// CHECK: [[MAP4:#.*]] = affine_map<(d0, d1, d2) -> (d0, (d1 floordiv 8) mod 2, (d1 floordiv 4) mod 2, d1 mod 4, d2)>
// CHECK: [[MAP5:#.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: @reshape_collabse_dims
func @reshape_collabse_dims(%arg0 : tensor<1x2x2x4x3xf32>) -> tensor<1x16x3xf32> {
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<1x2x2x4x3xf32>) -> tensor<1x16x3xf32>
  return %0 : tensor<1x16x3xf32>
}
// CHECK: linalg.generic {
// CHECK-SAME: args_in = 1
// CHECK-SAME: args_out = 1
// CHECK-SAME: indexing_maps
// CHECK-SAME: [[MAP4]], [[MAP5]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-NEXT: ^{{.+}}([[ARG:%.*]]: f32)
// CHECK-NEXT: linalg.yield [[ARG]] : f32

// -----

// CHECK: [[MAP6:#.*]] = affine_map<(d0) -> ((d0 floordiv 8) mod 3, (d0 floordiv 4) mod 2, d0 mod 4)>
// CHECK: [[MAP7:#.*]] = affine_map<(d0) -> (d0)>
// CHECK: @reshape_collabse_single_dim
func @reshape_collabse_single_dim(%arg0 : tensor<3x2x4xf32>) -> tensor<24xf32> {
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<3x2x4xf32>) -> tensor<24xf32>
  return %0 : tensor<24xf32>
}
// CHECK: linalg.generic {
// CHECK-SAME: args_in = 1
// CHECK-SAME: args_out = 1
// CHECK-SAME: indexing_maps
// CHECK-SAME: [[MAP6]], [[MAP7]]
// CHECK-SAME: iterator_types = ["parallel"]}
// CHECK-NEXT: ^{{.+}}([[ARG:%.*]]: f32)
// CHECK-NEXT: linalg.yield [[ARG]] : f32

// -----

// CHECK: [[MAP8:#.*]] = affine_map<(d0, d1) -> (d0, (d1 floordiv 8) mod 8, d1 mod 8, 0)>
// CHECK: [[MAP9:#.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: @reshape_collabse_single_dim_inner
func @reshape_collabse_single_dim_inner(%arg0 : tensor<1x8x8x1xf32>) -> tensor<1x64xf32> {
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<1x8x8x1xf32>) -> tensor<1x64xf32>
  return %0 : tensor<1x64xf32>
}
// CHECK: linalg.generic {
// CHECK-SAME: args_in = 1
// CHECK-SAME: args_out = 1
// CHECK-SAME: indexing_maps
// CHECK-SAME: [[MAP8]], [[MAP9]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-NEXT: ^{{.+}}([[ARG:%.*]]: f32)
// CHECK-NEXT: linalg.yield [[ARG]] : f32
