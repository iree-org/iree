// RUN: iree-opt --split-input-file --verify-diagnostics --iree-dispatch-creation-transpose-generic-ops -canonicalize -cse --mlir-print-local-scope %s | FileCheck %s

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
// CHECK-LABEL: func public @supported_conv(
//       CHECK:   %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>]
//  CHECK-SAME:       ins(%[[CONV]] :
//       CHECK:   return %[[GENERIC]]

// -----

util.func @generalize_to_any_linalg_op(%arg0 : tensor<?x?x?x?xi8>, %arg1 : tensor<?x?x?x?xi8>,
    %arg2 : tensor<?x?x?x?xi64>, %arg3 : tensor<?x?x?x?xi64>, %arg4 : tensor<?x?x?x?xi8>) -> tensor<?x?x?x?xi8> {
  %c0_i64 = arith.constant 0 : i64
  %0 = linalg.conv_2d_nhwc_hwcf_q {
      dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%arg0, %arg1, %c0_i64, %c0_i64 : tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>, i64, i64)
      outs(%arg2 : tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%0 : tensor<?x?x?x?xi64>) outs(%arg4 : tensor<?x?x?x?xi8>) {
  ^bb0(%in: i64, %out: i8):
    %3 = arith.trunci %in : i64 to i32
    %4 = arith.sitofp %3 : i32 to f32
    %5 = arith.fptosi %4 : f32 to i8
    linalg.yield %5 : i8
  } -> tensor<?x?x?x?xi8>
  util.return %2 : tensor<?x?x?x?xi8>
}
// CHECK-LABEL: func public @generalize_to_any_linalg_op(
//       CHECK:   %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf_q
//       CHECK:   %[[RESULT:.+]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>]
//       CHECK:   return %[[RESULT]]

//  -----

//      CHECK: util.func public @interchange
//      CHECK:   linalg.generic {indexing_maps = [
// CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
// CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d3, d0, d1)>
// CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d2, d0, d1)>
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
util.func public @interchange(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) {
  %0 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
    affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>],
    iterator_types = ["reduction", "parallel", "parallel", "parallel"]}
  ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  outs(%arg2 : tensor<?x?x?xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    %m = arith.mulf %arg3, %arg4 : f32
    %a = arith.addf %arg5, %m : f32
    linalg.yield %a : f32
  } -> tensor<?x?x?xf32>
  util.return %0 : tensor<?x?x?xf32>
}
