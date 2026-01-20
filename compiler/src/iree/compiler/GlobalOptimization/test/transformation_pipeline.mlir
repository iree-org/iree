// RUN: iree-opt --split-input-file --iree-global-optimization-transformation-pipeline %s | FileCheck %s

// CHECK-LABEL: @empty
util.func public @empty() {
  // CHECK-NEXT: util.return
  util.return
}

// -----

util.func public @elementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  %1 = arith.subf %0, %arg0 : tensor<4xf32>
  %2 = arith.mulf %1, %arg0 : tensor<4xf32>
  util.return %2 : tensor<4xf32>
}

// CHECK-LABEL: util.func public @elementwiseOps(%arg0: tensor<4xf32>) -> tensor<4xf32> {
//       CHECK:   %{{.+}} = linalg.generic
//       CHECK:     %{{.+}} = arith.addf %{{.+}}, %{{.+}} : f32
//       CHECK:   %{{.+}} = linalg.generic
//       CHECK:     %{{.+}} = arith.subf %{{.+}}, %{{.+}} : f32
//       CHECK:   %{{.+}} = linalg.generic
//       CHECK:     %{{.+}} = arith.mulf %{{.+}}, %{{.+}} : f32

// -----

// Test that transposes get fused with the strided convolution.
util.func public @transpose_with_strided_conv(%arg0: tensor<40x1x1x32xbf16>, %arg1: tensor<16x192x128x32xbf16>) -> tensor<16x96x64x40xbf16> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<16x32x192x128xbf16>
  %transposed = linalg.transpose ins(%arg1 : tensor<16x192x128x32xbf16>) outs(%0 : tensor<16x32x192x128xbf16>) permutation = [0, 3, 1, 2]
  %1 = tensor.empty() : tensor<40x32x1x1xbf16>
  %transposed_0 = linalg.transpose ins(%arg0 : tensor<40x1x1x32xbf16>) outs(%1 : tensor<40x32x1x1xbf16>) permutation = [0, 3, 1, 2]
  %2 = tensor.empty() : tensor<16x40x96x64xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<16x40x96x64xf32>) -> tensor<16x40x96x64xf32>
  %4 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%transposed, %transposed_0 : tensor<16x32x192x128xbf16>, tensor<40x32x1x1xbf16>) outs(%3 : tensor<16x40x96x64xf32>) -> tensor<16x40x96x64xf32>
  %5 = tensor.empty() : tensor<16x40x96x64xbf16>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<16x40x96x64xf32>) outs(%5 : tensor<16x40x96x64xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %9 = arith.truncf %in : f32 to bf16
    linalg.yield %9 : bf16
  } -> tensor<16x40x96x64xbf16>
  %7 = tensor.empty() : tensor<16x96x64x40xbf16>
  %transposed_1 = linalg.transpose ins(%6 : tensor<16x40x96x64xbf16>) outs(%7 : tensor<16x96x64x40xbf16>) permutation = [0, 2, 3, 1]
  util.return %transposed_1 : tensor<16x96x64x40xbf16>
}

// CHECK-LABEL: util.func public @transpose_with_strided_conv
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<40x1x1x32xbf16>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<16x192x128x32xbf16>
//       CHECK:   %[[START0:.+]] = iree_tensor_ext.compute_barrier.start %[[ARG1]]
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]]
//       CHECK:   %[[START1:.+]] = iree_tensor_ext.compute_barrier.start %[[COLLAPSED]]
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[START0]][0, 0, 0, 0] [16, 96, 64, 32] [1, 2, 2, 1]
//  CHECK-SAME:     tensor<16x192x128x32xbf16> to tensor<16x96x64x32xbf16>
//       CHECK:   %[[CONTRACT:.+]] = linalg.generic
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
//  CHECK-SAME:     ins(%[[SLICE]], %[[START1]] : tensor<16x96x64x32xbf16>, tensor<40x32xbf16>)
//       CHECK:   %[[TRUNC:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[CONTRACT]] : tensor<16x96x64x40xf32>)
//       CHECK:   %[[END:.+]] = iree_tensor_ext.compute_barrier.end %[[TRUNC]]
//       CHECK:   util.return %[[END]]
