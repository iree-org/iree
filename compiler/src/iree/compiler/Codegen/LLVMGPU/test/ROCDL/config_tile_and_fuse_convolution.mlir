// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx1100 \
// RUN:   --iree-codegen-llvmgpu-use-igemm=false --iree-codegen-llvmgpu-use-direct-convolution=false \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

#input_map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 * 2 + d5, d3 * 2 + d6)>
#filter_map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#output_map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

func.func @generic_dynamic_nchw_fchw_conv(
    %input: tensor<?x3x230x230xf32>, %filter: tensor<64x3x7x7xf32>,
    %init: tensor<?x64x112x112xf32>) -> tensor<?x64x112x112xf32> {
  // CHECK-LABEL: func.func @generic_dynamic_nchw_fchw_conv
  // CHECK:       #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse>
  // CHECK:       linalg.generic
  // CHECK-NOT:   promote_operands
  // CHECK:       ^bb0
  %result = linalg.generic {
      indexing_maps = [#input_map, #filter_map, #output_map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
      ins(%input, %filter : tensor<?x3x230x230xf32>, tensor<64x3x7x7xf32>)
      outs(%init : tensor<?x64x112x112xf32>) {
  ^bb0(%in: f32, %weight: f32, %out: f32):
    %product = arith.mulf %in, %weight : f32
    %sum = arith.addf %out, %product : f32
    linalg.yield %sum : f32
  } -> tensor<?x64x112x112xf32>
  return %result : tensor<?x64x112x112xf32>
}
