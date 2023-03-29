// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-flow-form-dispatch-regions{fuse-pad-with-consumers}))" --split-input-file %s | FileCheck %s

func.func @fuse_with_consumer(%arg0 : tensor<?x?x?x?xf32>, %arg1 : index,
    %arg2 : index, %arg3 : index, %arg4 : index,
    %arg5 : tensor<?x?x?x?xf32>, %arg6 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %cst = arith.constant 42.0 : f32
  %0 = tensor.pad %arg0 low[0, 0, 0, 0] high[%arg1, %arg2, %arg3, %arg4] {
  ^bb0(%b0 : index, %b1 : index, %b2 : index, %b3 : index) :
    tensor.yield %cst : f32
  } : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32>
  %1 = linalg.conv_2d_nhwc_hwcf ins(%0, %arg5 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%arg6 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}
// CHECK-LABEL: func @fuse_with_consumer
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG5:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG6:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
//       CHECK:   %[[RETURN:.+]] = flow.dispatch.region
//       CHECK:     %[[PADDED:.+]] = tensor.pad %[[ARG0]]
//       CHECK:     %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
//  CHECK-SAME:         ins(%[[PADDED]], %[[ARG5]] :
//  CHECK-SAME:         outs(%[[ARG6]] :
//       CHECK:     flow.return %[[CONV]]
//       CHECK:   return %[[RETURN]]
