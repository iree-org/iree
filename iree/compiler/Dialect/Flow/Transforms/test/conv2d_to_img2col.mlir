// RUN: iree-opt -split-input-file -iree-flow-convert-conv2d-to-img2col %s | IreeFileCheck %s


func @conv_16433136(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
    %0 = linalg.conv_2d_input_nhwc_filter_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%arg0, %arg1: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
      outs(%arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
    return %0 : tensor<1x14x14x16xf32>
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d3, d2 + d4, d5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//      CHECK: @conv_16433136
//      CHECK: %[[INPUT:.+]]: tensor<1x16x16x4xf32>
//      CHECK: %[[FILTER:.+]]: tensor<3x3x4x16xf32>
//      CHECK: %[[OUTPUT:.+]]: tensor<1x14x14x16xf32>
//      CHECK: %[[INIT_COL_TENSOR:.+]] = linalg.init_tensor [1, 14, 14, 3, 3, 4] : tensor<1x14x14x3x3x4xf32>
//      CHECK: %[[COL_TENSOR:.+]] = linalg.generic
//           CHECK-SAME: #[[MAP0]]
//           CHECK-SAME: #[[MAP1]]
//                CHECK: ^bb0(%[[IN_DATA:.+]]: f32, %[[OUT_DATA:.+]]: f32)
//                CHECK: linalg.yield %[[IN_DATA]] : f32
//      CHECK-DAG: %[[RESHAPED_INIT_COL_TENSOR:.+]] = linalg.tensor_collapse_shape %[[COL_TENSOR]]
//           CHECK-SAME: [0, 1, 2], [3, 4, 5]
//           CHECK-SAME: tensor<1x14x14x3x3x4xf32> into tensor<196x36xf32>
//      CHECK-DAG: %[[RESHAPED_FILTER:.+]] = linalg.tensor_collapse_shape %[[FILTER]]
//           CHECK-SAME: [0, 1, 2], [3]
//           CHECK-SAME: tensor<3x3x4x16xf32> into tensor<36x16xf32>
//      CHECK-DAG: %[[RESHAPED_OUTPUT:.+]] = linalg.tensor_collapse_shape %[[OUTPUT]]
//           CHECK-SAME: [0, 1, 2], [3]
//      CHECK: %[[MATMUL_RESULT:.+]] = linalg.matmul ins(%[[RESHAPED_INIT_COL_TENSOR]], %[[RESHAPED_FILTER]] : tensor<196x36xf32>, tensor<36x16xf32>) outs(%[[RESHAPED_OUTPUT]] : tensor<196x16xf32>)
//      CHECK: %[[RESULT:.+]] = linalg.tensor_expand_shape %[[MATMUL_RESULT]] {{\[}}[0, 1, 2], [3]] : tensor<196x16xf32> into tensor<1x14x14x16xf32>
//      CHECK: return %[[RESULT]]
