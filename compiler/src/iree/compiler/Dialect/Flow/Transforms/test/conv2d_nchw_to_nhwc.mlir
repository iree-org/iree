// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(func.func(iree-flow-convert-conv-nchw-to-nhwc))" %s | FileCheck %s

func.func @batch_conv(%arg0: tensor<8x4x16x16xf32>, %arg1: tensor<16x4x3x3xf32>, %arg2: tensor<8x16x14x14xf32>) -> tensor<8x16x14x14xf32> {
    %0 = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%arg0, %arg1: tensor<8x4x16x16xf32>, tensor<16x4x3x3xf32>)
      outs(%arg2: tensor<8x16x14x14xf32>) -> tensor<8x16x14x14xf32>
    return %0 : tensor<8x16x14x14xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1, d0)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
//      CHECK: @batch_conv
//      CHECK: %[[INPUT:.+]]: tensor<8x4x16x16xf32>
//      CHECK: %[[FILTER:.+]]: tensor<16x4x3x3xf32>
//      CHECK: %[[OUTPUT:.+]]: tensor<8x16x14x14xf32>
//      CHECK: %[[INIT_INPUT_TRANSPOSE:.+]] = tensor.empty() {__nchw_to_nhwc_init__} : tensor<8x16x16x4xf32>
//      CHECK: %[[TRANSPOSED_INPUT:.+]] = linalg.generic
//           CHECK-SAME: #[[MAP0]]
//           CHECK-SAME: #[[MAP1]]
//           CHECK-SAME: ins(%[[INPUT]] : tensor<8x4x16x16xf32>) outs(%[[INIT_INPUT_TRANSPOSE]] : tensor<8x16x16x4xf32>)
//      CHECK: %[[INIT_FILTER_TRANSPOSE:.+]] = tensor.empty() {__nchw_to_nhwc_init__} : tensor<3x3x4x16xf32>
//      CHECK: %[[TRANSPOSED_FILTER:.+]] = linalg.generic
//           CHECK-SAME: #[[MAP0]]
//           CHECK-SAME: #[[MAP2]]
//           CHECK-SAME: ins(%[[FILTER]] : tensor<16x4x3x3xf32>) outs(%[[INIT_FILTER_TRANSPOSE]] : tensor<3x3x4x16xf32>)
//      CHECK: %[[INIT_OUTPUT_TRANSPOSE:.+]] = tensor.empty() {__nchw_to_nhwc_init__} : tensor<8x14x14x16xf32>
//      CHECK: %[[TRANSPOSED_OUTPUT:.+]] = linalg.generic
//           CHECK-SAME: #[[MAP0]]
//           CHECK-SAME: #[[MAP1]]
//           CHECK-SAME: ins(%[[OUTPUT]] : tensor<8x16x14x14xf32>) outs(%[[INIT_OUTPUT_TRANSPOSE]] : tensor<8x14x14x16xf32>)
//      CHECK: %[[TRANSPOSED_RESULT:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%[[TRANSPOSED_INPUT]], %[[TRANSPOSED_FILTER]] : tensor<8x16x16x4xf32>, tensor<3x3x4x16xf32>) outs(%[[TRANSPOSED_OUTPUT]] : tensor<8x14x14x16xf32>) -> tensor<8x14x14x16xf32>
//      CHECK: %[[INIT_RESULT:.+]] = tensor.empty() {__nchw_to_nhwc_init__} : tensor<8x16x14x14xf32>
//      CHECK: %[[RESULT:.+]] = linalg.generic
//           CHECK-SAME: #[[MAP0]]
//           CHECK-SAME: #[[MAP3]]
//           CHECK-SAME: ins(%[[TRANSPOSED_RESULT]] : tensor<8x14x14x16xf32>) outs(%[[INIT_RESULT]] : tensor<8x16x14x14xf32>)
//      CHECK: return %[[RESULT]] : tensor<8x16x14x14xf32>
