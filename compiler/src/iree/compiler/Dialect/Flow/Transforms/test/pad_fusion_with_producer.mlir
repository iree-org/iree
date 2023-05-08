// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-flow-form-dispatch-regions{fuse-pad-with-producers}))" --split-input-file %s | FileCheck %s

func.func @fuse_pad_with_producer(%arg0 : tensor<?x?x?x?xf32>,
    %arg1 : tensor<?x?x?x?xf32>, %arg2 : tensor<?x?x?x?xf32>,
    %arg3 : tensor<?xf32>, %arg4 : index, %arg5 : index, %arg6 : index,
    %arg7 : index) -> tensor<?x?x?x?xf32> {
  %cst = arith.constant 42.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %d0 = tensor.dim %arg2, %c0 : tensor<?x?x?x?xf32>
  %d1 = tensor.dim %arg2, %c1 : tensor<?x?x?x?xf32>
  %d2 = tensor.dim %arg2, %c2 : tensor<?x?x?x?xf32>
  %d3 = tensor.dim %arg2, %c3 : tensor<?x?x?x?xf32>
  %0 = linalg.conv_2d_nhwc_hwcf
      ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1 = tensor.empty(%d0, %d1, %d2, %d3) : tensor<?x?x?x?xf32>
  %2 = linalg.generic {
      indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%0, %arg3 : tensor<?x?x?x?xf32>, tensor<?xf32>)
      outs(%1 : tensor<?x?x?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %3 = arith.addf %b0, %b1 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?x?x?xf32>
  %4 = tensor.pad %2 low[0, 0, 0, 0] high[%arg4, %arg5, %arg6, %arg7] {
  ^bb0(%b0 : index, %b1 : index, %b2 : index, %b3 : index) :
    tensor.yield %cst : f32
  } : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32>
  return %4 : tensor<?x?x?x?xf32>
}
// CHECK-LABEL: func @fuse_pad_with_producer(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<?xf32>
//       CHECK:   %[[RETURN:.+]] = flow.dispatch.region
//       CHECK:     %[[CONV:[a-zA-Z0-9]+]] = linalg.conv_2d_nhwc_hwcf
//  CHECK-SAME:         ins(%[[ARG0]], %[[ARG1]] :
//  CHECK-SAME:         outs(%[[ARG2]] :
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[CONV]], %[[ARG3]]
//       CHECK:     %[[PADDED:.+]] = tensor.pad %[[GENERIC]]
//       CHECK:     flow.return %[[PADDED]]
//       CHEKC:   return %[[RETURN]]
