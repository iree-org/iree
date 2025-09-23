// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{fuse-pad-with-consumers}))" --split-input-file %s | FileCheck %s

util.func public @fuse_with_consumer_named_op(%arg0 : tensor<?x?x?x?xf32>, %arg1 : index,
    %arg2 : index, %arg3 : index, %arg4 : index,
    %arg5 : tensor<?x?x?x?xf32>, %arg6 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %cst = arith.constant 42.0 : f32
  %0 = tensor.pad %arg0 low[0, 0, 0, 0] high[%arg1, %arg2, %arg3, %arg4] {
  ^bb0(%b0 : index, %b1 : index, %b2 : index, %b3 : index) :
    tensor.yield %cst : f32
  } : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32>
  %1 = linalg.conv_2d_nhwc_hwcf ins(%0, %arg5 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%arg6 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  util.return %1 : tensor<?x?x?x?xf32>
}
// CHECK-LABEL: util.func public @fuse_with_consumer_named_op
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG5:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG6:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//       CHECK:   %[[RETURN:.+]] = flow.dispatch.region
//       CHECK:     %[[PADDED:.+]] = tensor.pad %[[ARG0]]
//       CHECK:     %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
//  CHECK-SAME:         ins(%[[PADDED]], %[[ARG5]] :
//  CHECK-SAME:         outs(%[[ARG6]] :
//       CHECK:     flow.return %[[CONV]]
//       CHECK:   util.return %[[RETURN]]

// -----

util.func public @fuse_with_consumer_generalized(%arg0: tensor<?x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: tensor<?x?x?x?xf32>, %arg6: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %cst = arith.constant 4.200000e+01 : f32
  %padded = tensor.pad %arg0 low[0, 0, 0, 0] high[%arg1, %arg2, %arg3, %arg4] {
  ^bb0(%arg7: index, %arg8: index, %arg9: index, %arg10: index):
    tensor.yield %cst : f32
  } : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%padded, %arg5 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%arg6 : tensor<?x?x?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32                                                                                                                                                                     } -> tensor<?x?x?x?xf32>
  util.return %0 : tensor<?x?x?x?xf32>
}
// CHECK-LABEL: util.func public @fuse_with_consumer_generalized
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG5:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG6:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//       CHECK:   %[[RETURN:.+]] = flow.dispatch.region
//       CHECK:     %[[PADDED:.+]] = tensor.pad %[[ARG0]]
//       CHECK:     %[[CONV:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[PADDED]], %[[ARG5]] :
//  CHECK-SAME:         outs(%[[ARG6]] :
//       CHECK:     flow.return %[[CONV]]
//       CHECK:   util.return %[[RETURN]]
