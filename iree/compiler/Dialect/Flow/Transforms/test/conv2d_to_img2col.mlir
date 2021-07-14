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

// -----

func @depthwise_conv_hwc_114x16x3(%input: tensor<1x114x114x16xf32>, %filter: tensor<3x3x16xf32>, %output: tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32> {
    %0 = linalg.depthwise_conv_2d_input_nhwc_filter_hwc {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x114x114x16xf32>, tensor<3x3x16xf32>) outs(%output : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
    return %0 : tensor<1x112x112x16xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 + d4, d3 + d5)>
// CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-DAG: #[[MAP6:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
// CHECK: @depthwise_conv_hwc_114x16x3
// CHECK-SAME: %[[INPUT:.+]]: tensor<1x114x114x16xf32>
// CHECK-SAME: %[[FILTER:.+]]: tensor<3x3x16xf32>
// CHECK-SAME: %[[OUTPUT:.+]]: tensor<1x112x112x16xf32>
//      CHECK: %[[INPUT_T_INIT:.+]] = linalg.init_tensor [1, 16, 114, 114] : tensor<1x16x114x114xf32>
//      CHECK: %[[INPUT_T:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[INPUT]] : tensor<1x114x114x16xf32>) outs(%[[INPUT_T_INIT]] : tensor<1x16x114x114xf32>) {
// CHECK-NEXT: ^bb0(%arg3: f32, %arg4: f32):
// CHECK-NEXT:     linalg.yield %arg3 : f32
// CHECK-NEXT:  } -> tensor<1x16x114x114xf32>
//      CHECK: %[[FILTER_T_INIT:.+]] = linalg.init_tensor [16, 3, 3] : tensor<16x3x3xf32>
//      CHECK: %[[FILTER_T:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP2]], #[[MAP3]]
// CHECK-SMAE: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[FILTER]] : tensor<3x3x16xf32>) outs(%[[FILTER_T_INIT]] : tensor<16x3x3xf32>) {
// CHECK-NEXT:      ^bb0(%{{.*}}: f32, %{{.*}}: f32):
//      CHECK:      linalg.yield
//      CHECK:    } -> tensor<16x3x3xf32>
//      CHECK: %[[INIT_OUTPUT_TENSOR:.+]] = linalg.init_tensor [1, 16, 112, 112] : tensor<1x16x112x112xf32>
//      CHECK: %[[OUTPUT_T:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[OUTPUT]] : tensor<1x112x112x16xf32>) outs(%[[INIT_OUTPUT_TENSOR]] : tensor<1x16x112x112xf32>) {
// CHECK-NEXT:  ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:     linalg.yield
// CHECK-NEXT:  } -> tensor<1x16x112x112xf32>
//      CHECK:  %[[INIT_COL_TENSOR:.+]] = linalg.init_tensor [1, 16, 112, 112, 3, 3] : tensor<1x16x112x112x3x3xf32>
//      CHECK: %[[COL_TENSOR:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP4]], #[[MAP5]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:   ins(%[[INPUT_T]] : tensor<1x16x114x114xf32>) outs(%[[INIT_COL_TENSOR]] : tensor<1x16x112x112x3x3xf32>) {
// CHECK-NEXT:      ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:         linalg.yield
// CHECK-NEXT:    } -> tensor<1x16x112x112x3x3xf32>
//      CHECK: %[[COL_TENSOR_R:.+]] = linalg.tensor_collapse_shape %[[COL_TENSOR]]
// CHECK-SAME:    tensor<1x16x112x112x3x3xf32> into tensor<16x12544x9xf32>
//      CHECK: %[[FILTER_T_R:.+]] = linalg.tensor_collapse_shape %[[FILTER_T]]
// CHECK-SAME:    tensor<16x3x3xf32> into tensor<16x9xf32>
//      CHECK: %[[OUTPUT_T_R:.+]] = linalg.tensor_collapse_shape %[[OUTPUT_T]]
// CHECK-SAME:    tensor<1x16x112x112xf32> into tensor<16x12544xf32>
//      CHECK: %[[BMV_RESULT:.+]] = linalg.batch_matvec ins(%[[COL_TENSOR_R]], %[[FILTER_T_R]] : tensor<16x12544x9xf32>, tensor<16x9xf32>) outs(%[[OUTPUT_T_R]] : tensor<16x12544xf32>) -> tensor<16x12544xf32>
//      CHECK: %[[RESULT_R:.+]] = linalg.tensor_expand_shape %[[BMV_RESULT]]
// CHECK-SAME:    tensor<16x12544xf32> into tensor<1x16x112x112xf32>
//      CHECK: %[[RESULT_INIT:.+]] = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
//      CHECK: %[[RESULT:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP6]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[RESULT_R]] : tensor<1x16x112x112xf32>) outs(%[[RESULT_INIT]] : tensor<1x112x112x16xf32>) {
// CHECK-NEXT:      ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:      linalg.yield
// CHECK-NEXT:    } -> tensor<1x112x112x16xf32>
//      CHECK: return %[[RESULT]] : tensor<1x112x112x16xf32>
