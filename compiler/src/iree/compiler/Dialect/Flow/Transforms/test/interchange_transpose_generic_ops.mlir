// RUN: iree-opt --split-input-file --verify-diagnostics --iree-flow-interchange-transpose-generic-ops --canonicalize -cse --mlir-print-local-scope %s | FileCheck %s

util.func @supported_conv(%arg0 : tensor<2x130x130x16xf16>, %arg1 : tensor<3x3x16x320xf16>) -> tensor<2x320x128x128xf16> {
  %empty = tensor.empty() : tensor<2x128x128x320xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%arg0, %arg1 : tensor<2x130x130x16xf16>, tensor<3x3x16x320xf16>)
      outs(%fill : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
  %empty1 = tensor.empty() : tensor<2x320x128x128xf16>
  %truncf = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv : tensor<2x128x128x320xf32>) outs(%empty1 : tensor<2x320x128x128xf16>) {
    ^bb0(%b0 : f32, %b1 :f16):
      %0 = arith.truncf %b0 : f32 to f16
      linalg.yield %0 : f16
  } -> tensor<2x320x128x128xf16>
  util.return %truncf : tensor<2x320x128x128xf16>
}
// CHECK-LABEL: func public @supported_conv
//       CHECK:   %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>]
//  CHECK-SAME:       ins(%[[CONV]] :
//       CHECK:   return %[[GENERIC]]
