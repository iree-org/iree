// RUN: iree-opt --mlir-print-local-scope --split-input-file \
// RUN:   --iree-gpu-test-target=gfx942 \
// RUN:   --iree-codegen-llvmgpu-use-vector-distribution \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s \
// RUN:   | FileCheck %s

// Verify that reductions fused with multi-output generics that have transposed
// outputs select #iree_gpu.pipeline<VectorDistribute> and attach lowering configs to the
// parallel op with the transposed output.

// 2D case: reduction over dim 1, elementwise with (d0, d1) -> (d1, d0) output.

// CHECK-LABEL: func.func @reduction_2d_transposed_output
//  CHECK-SAME:   pipeline = #iree_gpu.pipeline<VectorDistribute>
//       CHECK:   linalg.generic {{.*}} iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:     lowering_config
//       CHECK:   linalg.generic {{.*}} iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:     lowering_config

func.func @reduction_2d_transposed_output(
    %input: tensor<512x4096xf32>,
    %result: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x4096xf32>>,
    %result_t: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x512xf32>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %empty_red = tensor.empty() : tensor<512xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%empty_red : tensor<512xf32>) -> tensor<512xf32>
  %red = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<512x4096xf32>) outs(%filled : tensor<512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sq = arith.mulf %in, %in : f32
    %add = arith.addf %sq, %out : f32
    linalg.yield %add : f32
  } -> tensor<512xf32>
  %empty0 = tensor.empty() : tensor<512x4096xf32>
  %empty1 = tensor.empty() : tensor<4096x512xf32>
  %res:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%input, %red : tensor<512x4096xf32>, tensor<512xf32>)
      outs(%empty0, %empty1 : tensor<512x4096xf32>, tensor<4096x512xf32>) {
  ^bb0(%in: f32, %r: f32, %o0: f32, %o1: f32):
    %v = arith.mulf %in, %r : f32
    linalg.yield %v, %v : f32, f32
  } -> (tensor<512x4096xf32>, tensor<4096x512xf32>)
  iree_tensor_ext.dispatch.tensor.store %res#0, %result, offsets = [0, 0], sizes = [512, 4096], strides = [1, 1] : tensor<512x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x4096xf32>>
  iree_tensor_ext.dispatch.tensor.store %res#1, %result_t, offsets = [0, 0], sizes = [4096, 512], strides = [1, 1] : tensor<4096x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x512xf32>>
  return
}

// -----

// 3D case: reduction over dim 2, elementwise with (d0, d1, d2) -> (d0, d2, d1) output.

// CHECK-LABEL: func.func @reduction_3d_transposed_output
//  CHECK-SAME:   pipeline = #iree_gpu.pipeline<VectorDistribute>
//       CHECK:   linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:     lowering_config
//       CHECK:   linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:     lowering_config

func.func @reduction_3d_transposed_output(
    %input: tensor<16x32x4096xf32>,
    %result: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x32x4096xf32>>,
    %result_t: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x4096x32xf32>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %empty_red = tensor.empty() : tensor<16x32xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%empty_red : tensor<16x32xf32>) -> tensor<16x32xf32>
  %red = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%input : tensor<16x32x4096xf32>) outs(%filled : tensor<16x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sq = arith.mulf %in, %in : f32
    %add = arith.addf %sq, %out : f32
    linalg.yield %add : f32
  } -> tensor<16x32xf32>
  %empty0 = tensor.empty() : tensor<16x32x4096xf32>
  %empty1 = tensor.empty() : tensor<16x4096x32xf32>
  %res:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d2, d1)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%input, %red : tensor<16x32x4096xf32>, tensor<16x32xf32>)
      outs(%empty0, %empty1 : tensor<16x32x4096xf32>, tensor<16x4096x32xf32>) {
  ^bb0(%in: f32, %r: f32, %o0: f32, %o1: f32):
    %v = arith.mulf %in, %r : f32
    linalg.yield %v, %v : f32, f32
  } -> (tensor<16x32x4096xf32>, tensor<16x4096x32xf32>)
  iree_tensor_ext.dispatch.tensor.store %res#0, %result, offsets = [0, 0, 0], sizes = [16, 32, 4096], strides = [1, 1, 1] : tensor<16x32x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x32x4096xf32>>
  iree_tensor_ext.dispatch.tensor.store %res#1, %result_t, offsets = [0, 0, 0], sizes = [16, 4096, 32], strides = [1, 1, 1] : tensor<16x4096x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x4096x32xf32>>
  return
}
