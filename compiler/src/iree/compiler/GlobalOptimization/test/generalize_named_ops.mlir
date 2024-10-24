// RUN: iree-opt --mlir-print-local-scope --pass-pipeline="builtin.module(util.func(iree-global-opt-generalize-linalg-named-ops))" --split-input-file %s | FileCheck %s

util.func public @generalize_op(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %empty = tensor.empty(%d0, %d1): tensor<?x?xf32>
  %add = linalg.add ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %add : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @generalize_op
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//       CHECK:   util.return %[[GENERIC]]

// -----

util.func public @no_generalize_op_within_dispatch(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %dispatch = flow.dispatch.region[] -> (tensor<?x?xf32>{%d0, %d1}) {
    %empty = tensor.empty(%d0, %d1): tensor<?x?xf32>
    %add = linalg.add ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    flow.return %add : tensor<?x?xf32>
  }
  util.return %dispatch : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @no_generalize_op_within_dispatch
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[ADD:.+]] = linalg.add
//       CHECK:     flow.return %[[ADD]]
//       CHECK:   util.return %[[DISPATCH]]

// -----

util.func public @generalize_1x1_nhwc_conv_2d(%input: tensor<1x4x?x2xf32>, %filter: tensor<1x1x2x7xf32>) -> tensor<1x4x?x7xf32> {
    %c2 = arith.constant 2 : index
    %d2 = tensor.dim %input, %c2 : tensor<1x4x?x2xf32>
    %0 = tensor.empty(%d2) : tensor<1x4x?x7xf32>
    %1 = linalg.conv_2d_nhwc_hwcf {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x4x?x2xf32>, tensor<1x1x2x7xf32>) outs(%0 : tensor<1x4x?x7xf32>) -> tensor<1x4x?x7xf32>
    util.return %1 : tensor<1x4x?x7xf32>
}

// CHECK-LABEL: @generalize_1x1_nhwc_conv_2d
//       CHECK:   %[[RESULT:.*]] = linalg.generic
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @generalize_1x1_nchw_conv_2d(%input: tensor<1x2x4x5xf32>, %filter: tensor<7x2x1x1xf32>) -> tensor<1x7x4x5xf32> {
    %0 = tensor.empty() : tensor<1x7x4x5xf32>
    %1 = linalg.conv_2d_nchw_fchw {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x2x4x5xf32>, tensor<7x2x1x1xf32>) outs(%0 : tensor<1x7x4x5xf32>) -> tensor<1x7x4x5xf32>
    util.return %1 : tensor<1x7x4x5xf32>
}

// CHECK-LABEL: @generalize_1x1_nchw_conv_2d
//       CHECK:   %[[RESULT:.*]] = linalg.generic
//       CHECK:   util.return %[[RESULT]]
