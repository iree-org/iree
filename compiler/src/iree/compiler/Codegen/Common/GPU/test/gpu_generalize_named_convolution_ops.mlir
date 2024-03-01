// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-generalize-named-convolution-and-contraction-ops))" %s | FileCheck %s

func.func @nhwc_convolution(%arg0: tensor<1x1x32x32xf16>, %arg1: tensor<1x1x32x128xf16>) -> tensor<1x1x32x128xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<1x1x32x128xf16>
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<1x1x32x128xf16>) -> tensor<1x1x32x128xf16>
  %2 = linalg.conv_2d_nhwc_hwcf {
    dilations = dense<1> : vector<2xi64>,
    strides = dense<1> : vector<2xi64>,
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 128, 1, 1, 32]]>
  }
         ins(%arg0, %arg1 : tensor<1x1x32x32xf16>, tensor<1x1x32x128xf16>)
         outs(%1 : tensor<1x1x32x128xf16>) -> tensor<1x1x32x128xf16>
  return %2 : tensor<1x1x32x128xf16>
}

//               CHECK: #[[$CONFIG:.+]] = #iree_codegen.lowering_config
// CHECK-SAME{LITERAL}: <tile_sizes = [[1, 1, 32, 128, 1, 1, 32]]>

// CHECK-LABEL: func.func @nhwc_convolution
//       CHECK:   linalg.generic
//  CHECK-SAME:     lowering_config = #[[$CONFIG]]

// -----

func.func @transpose_batch_matmul(%arg0: tensor<32x1x128xf16>, %arg1: tensor<32x?x128xf16>, %dim: index) -> tensor<32x1x?xf16> {
  %f0 = arith.constant 0.0 : f16
  %empty = tensor.empty(%dim) : tensor<32x1x?xf16>
  %fill = linalg.fill ins(%f0 : f16) outs(%empty : tensor<32x1x?xf16>) -> tensor<32x1x?xf16>
  %2 = linalg.batch_matmul_transpose_b ins(%arg0, %arg1 : tensor<32x1x128xf16>, tensor<32x?x128xf16>) outs(%fill : tensor<32x1x?xf16>) -> tensor<32x1x?xf16>
  return %2 : tensor<32x1x?xf16>
}

//       CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//       CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//       CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @transpose_batch_matmul
//       CHECK:  linalg.generic
//  CHECK-SAME:    indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
//       CHECK:  ^bb0(%[[A:.+]]: f16, %[[B:.+]]: f16, %[[OUT:.+]]: f16):
//       CHECK:    %[[MUL:.+]] = arith.mulf %[[A]], %[[B]] : f16
//       CHECK:    %[[ADD:.+]] = arith.addf %[[OUT]], %[[MUL]] : f16
//       CHECK:    linalg.yield %[[ADD]] : f16
