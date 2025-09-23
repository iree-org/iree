// RUN: iree-opt %s --mlir-print-local-scope --pass-pipeline='builtin.module(func.func(iree-codegen-gpu-tile-and-convert-conv-to-matmul, canonicalize, cse))' --split-input-file | FileCheck %s

#config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>}>
module {
  func.func @conv_generic(%a: tensor<1x3x66x8xf32>, %b: tensor<32x3x3x8xf32>, %c: tensor<1x1x64x32xf32>) -> tensor<1x1x64x32xf32> {
    %conv = linalg.generic {
      indexing_maps =
        [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
      ins(%a, %b : tensor<1x3x66x8xf32>, tensor<32x3x3x8xf32>) outs(%c : tensor<1x1x64x32xf32>)
      attrs = {lowering_config = #config} {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %14 = arith.mulf %in, %in_2 : f32
        %15 = arith.addf %out, %14 : f32
        linalg.yield %15 : f32
      } -> tensor<1x1x64x32xf32>
    return %conv : tensor<1x1x64x32xf32>
  }
}

// CHECK-LABEL: func.func @conv_generic
//       CHECK:  scf.for %{{.*}} = %c0 to %c3 step %c1
//       CHECK:    scf.for %{{.*}} = %c0 to %c3 step %c1
//       CHECK:      linalg.generic
//  CHECK-SAME:        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d6)

// -----

#config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>}>
module {
  func.func @conv_named_dilated(%a: tensor<1x5x68x8xf32>, %b: tensor<32x3x3x8xf32>, %c: tensor<1x1x64x32xf32>) -> tensor<1x1x64x32xf32> {
    %conv = linalg.conv_2d_nhwc_fhwc
      {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, lowering_config = #config}
      ins(%a, %b : tensor<1x5x68x8xf32>, tensor<32x3x3x8xf32>) outs(%c : tensor<1x1x64x32xf32>) -> tensor<1x1x64x32xf32>
    return %conv : tensor<1x1x64x32xf32>
  }
}

// CHECK-LABEL: func.func @conv_named_dilated
//       CHECK:  scf.for %{{.*}} = %c0 to %c3 step %c1
//       CHECK:    scf.for %{{.*}} = %c0 to %c3 step %c1
//       CHECK:      linalg.generic
//  CHECK-SAME:        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d6)
