// RUN: iree-opt --split-input-file --iree-dispatch-creation-tensor-pad-to-tensor-insert-slice --canonicalize %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-dispatch-creation-tensor-pad-to-tensor-insert-slice=skip-one-linalg-use-case --canonicalize %s | FileCheck %s --check-prefix=SKIP

util.func public @tensor_pad(%arg0 : tensor<?x?xf32>, %arg1 : tensor<f32>, %arg2 : index, %arg3 : index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %0 = tensor.extract %arg1[] : tensor<f32>
  %1 = tensor.pad %arg0 low[%c4, %arg2] high[%arg3, %c3]  {
  ^bb0(%arg4: index, %arg5: index):
    tensor.yield %0 : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  util.return %1 : tensor<?x?xf32>
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 4)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 3)>
//       CHECK: util.func public @tensor_pad
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<f32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//   CHECK-DAG:   %[[VAL:.+]] = tensor.extract %[[ARG1]]
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[RD0:.+]] = affine.apply #[[MAP0]]()[%[[D0]], %[[ARG3]]]
//   CHECK-DAG:   %[[RD1:.+]] = affine.apply #[[MAP1]]()[%[[D1]], %[[ARG2]]]
//       CHECK:   %[[INIT:.+]] = tensor.empty(%[[RD0]], %[[RD1]])
//       CHECK:   %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:       ins(%[[VAL]] :
//  CHECK-SAME:       outs(%[[INIT]] :
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//       CHECK:   %[[RESULT:.+]] = tensor.insert_slice %[[ARG0]] into %[[FILL]][4, %[[ARG2]]] [%[[D0]], %[[D1]]] [1, 1]
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @tensor_pad_static(%arg0: tensor<12x4xf32>, %arg1: tensor<f32>) -> tensor<18x12xf32> {
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %c3 = arith.constant 3 : index
  %0 = tensor.extract %arg1[] : tensor<f32>
  %1 = tensor.pad %arg0 low[%c4, %c5] high[%c2, %c3]  {
  ^bb0(%arg2: index, %arg3: index):
    tensor.yield %0 : f32
  } : tensor<12x4xf32> to tensor<18x12xf32>
  util.return %1 : tensor<18x12xf32>
}
// CHECK-LABEL: util.func public @tensor_pad_static
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<12x4xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<f32>
//   CHECK-DAG:   %[[VAL:.+]] = tensor.extract %[[ARG1]]
//       CHECK:   %[[INIT:.+]] = tensor.empty()
//       CHECK:   %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:       ins(%[[VAL]] :
//  CHECK-SAME:       outs(%[[INIT]] :
//       CHECK:   %[[RESULT:.+]] = tensor.insert_slice %[[ARG0]] into %[[FILL]][4, 5] [12, 4] [1, 1]
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @_main(%arg0: tensor<1x33x33x480xf32>, %arg1: tensor<3x3x480x1xf32>) -> tensor<1x33x33x480xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.pad %arg0 low[0, 4, 4, 0] high[0, 4, 4, 0] {
  ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
    tensor.yield %cst : f32
  } : tensor<1x33x33x480xf32> to tensor<1x41x41x480xf32>
  %1 = tensor.empty() : tensor<1x33x33x480xf32>
  %2 = tensor.collapse_shape %arg1 [[0], [1], [2, 3]] : tensor<3x3x480x1xf32> into tensor<3x3x480xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x33x33x480xf32>) -> tensor<1x33x33x480xf32>
  %4 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<4> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %2 : tensor<1x41x41x480xf32>, tensor<3x3x480xf32>) outs(%3 : tensor<1x33x33x480xf32>) -> tensor<1x33x33x480xf32>
  util.return %4 : tensor<1x33x33x480xf32>
}
// CHECK-NOT: tensor.pad
// SKIP: tensor.pad

// -----

#encoding = #iree_encoding.testing<>
util.func public @dispatch_dispatch_0_generic_512x1024_f32(
    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x1024xf32>>,
    %arg1: index, %arg2: index, %arg3: index, %arg4: index,
    %arg5: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 2 : index
  %1 = iree_tensor_ext.dispatch.workload.ordinal %arg4, 3 : index
  %2 = flow.dispatch.tie_shape %arg5 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>{%0, %1}
  %3 = iree_tensor_ext.dispatch.workload.ordinal %arg1, 0 : index
  %4 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 1 : index
  %5 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x1024xf32>> -> tensor<512x1024xf32>
  %padded = tensor.pad %5 low[0, 0] high[%3, %4] {
  ^bb0(%arg6: index, %arg7: index):
    tensor.yield %cst : f32
  } : tensor<512x1024xf32> to tensor<?x?xf32>
  %11 = iree_encoding.set_encoding %padded : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  iree_tensor_ext.dispatch.tensor.store %11, %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32, #encoding> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>{%0, %1}
  util.return
}
// CHECK:  #[[ENCODING:.+]] = #iree_encoding.testing<>
// CHECK:  util.func public @dispatch_dispatch_0_generic_512x1024_f32
// CHECK:    %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load
// CHECK:    %[[PAD:.+]] = tensor.pad %[[LOAD]] low
// CHECK:    %[[ENCODE:.+]] = iree_encoding.set_encoding %[[PAD]] : tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING]]>
// CHECK:    iree_tensor_ext.dispatch.tensor.store %[[ENCODE]],

// -----

// Do not break up pad within dispatches.

util.func @pad_within_dispatch(%arg0 : tensor<500x1000xf32>) -> tensor<512x1024xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = flow.dispatch.region -> (tensor<512x1024xf32>) {
    %1 = tensor.pad %arg0 low [0, 0] high[12, 24] {
    ^bb0(%arg1 : index, %arg2 : index):
      tensor.yield %cst : f32
    } : tensor<500x1000xf32> to tensor<512x1024xf32>
    flow.return %1 : tensor<512x1024xf32>
  }
  util.return %0 : tensor<512x1024xf32>
}
// CHECK-LABEL: func public @pad_within_dispatch
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[PAD:.+]] = tensor.pad
//       CHECK:     flow.return %[[PAD]]
//       CHECK:   return %[[DISPATCH]]
