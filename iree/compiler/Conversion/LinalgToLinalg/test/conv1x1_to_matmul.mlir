// RUN: iree-opt -split-input-file -iree-codegen-convert-conv2d-1x1-to-matmul %s | IreeFileCheck %s

func @conv_2d_1x1(%input: tensor<1x4x5x2xf32>, %filter: tensor<1x1x2x7xf32>) -> tensor<1x4x5x7xf32> {
    %0 = linalg.init_tensor [1, 4, 5, 7] : tensor<1x4x5x7xf32>
    %1 = linalg.conv_2d_input_nhwc_filter_hwcf {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x4x5x2xf32>, tensor<1x1x2x7xf32>) outs(%0 : tensor<1x4x5x7xf32>) -> tensor<1x4x5x7xf32>
    return %1 : tensor<1x4x5x7xf32>
}
// CHECK: @conv_2d_1x1
// CHECK: %[[INPUT:.+]]: tensor<1x4x5x2xf32>
// CHECK: %[[FILTER:.+]]: tensor<1x1x2x7xf32>
// CHECK: %[[OTUPUT:.+]] = linalg.init_tensor [1, 4, 5, 7] : tensor<1x4x5x7xf32>
// CHECK: %[[RESHAPED_INPUT:.+]] = linalg.tensor_collapse_shape %[[INPUT]] {{\[}}[0, 1, 2], [3]] : tensor<1x4x5x2xf32> into tensor<20x2xf32>
// CHECK: %[[RESHAPED_FILTER:.+]] = linalg.tensor_collapse_shape %[[FILTER]] {{\[}}[0, 1, 2], [3]] : tensor<1x1x2x7xf32> into tensor<2x7xf32>
// CHECK: %[[RESHAPED_OUTPUT:.+]] = linalg.tensor_collapse_shape %[[OTUPUT]] {{\[}}[0, 1, 2], [3]] : tensor<1x4x5x7xf32> into tensor<20x7xf32>
// CHECK: %[[MATMUL_RESULT:.+]] = linalg.matmul ins(%[[RESHAPED_INPUT]], %[[RESHAPED_FILTER]] : tensor<20x2xf32>, tensor<2x7xf32>) outs(%[[RESHAPED_OUTPUT]] : tensor<20x7xf32>)
// CHECK: %[[RESULT:.+]] = linalg.tensor_expand_shape %[[MATMUL_RESULT]] {{\[}}[0, 1, 2], [3]] : tensor<20x7xf32> into tensor<1x4x5x7xf32>
// CHECK: return %[[RESULT]]
