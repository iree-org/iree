// RUN: iree-opt --split-input-file -iree-flow-convert-conv2d-1x1-to-matmul %s | FileCheck %s

func.func @conv_2d_1x1(%input: tensor<1x4x5x2xf32>, %filter: tensor<1x1x2x7xf32>) -> tensor<1x4x5x7xf32> {
    %0 = linalg.init_tensor [1, 4, 5, 7] : tensor<1x4x5x7xf32>
    %1 = linalg.conv_2d_nhwc_hwcf {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x4x5x2xf32>, tensor<1x1x2x7xf32>) outs(%0 : tensor<1x4x5x7xf32>) -> tensor<1x4x5x7xf32>
    return %1 : tensor<1x4x5x7xf32>
}
// CHECK: @conv_2d_1x1
// CHECK: %[[INPUT:.+]]: tensor<1x4x5x2xf32>
// CHECK: %[[FILTER:.+]]: tensor<1x1x2x7xf32>
// CHECK: %[[OTUPUT:.+]] = linalg.init_tensor [1, 4, 5, 7] : tensor<1x4x5x7xf32>
// CHECK: %[[RESHAPED_INPUT:.+]] = tensor.collapse_shape %[[INPUT]] {{\[}}[0, 1, 2], [3]] : tensor<1x4x5x2xf32> into tensor<20x2xf32>
// CHECK: %[[RESHAPED_FILTER:.+]] = tensor.collapse_shape %[[FILTER]] {{\[}}[0, 1, 2], [3]] : tensor<1x1x2x7xf32> into tensor<2x7xf32>
// CHECK: %[[RESHAPED_OUTPUT:.+]] = tensor.collapse_shape %[[OTUPUT]] {{\[}}[0, 1, 2], [3]] : tensor<1x4x5x7xf32> into tensor<20x7xf32>
// CHECK: %[[MATMUL_RESULT:.+]] = linalg.matmul ins(%[[RESHAPED_INPUT]], %[[RESHAPED_FILTER]] : tensor<20x2xf32>, tensor<2x7xf32>) outs(%[[RESHAPED_OUTPUT]] : tensor<20x7xf32>)
// CHECK: %[[RESULT:.+]] = tensor.expand_shape %[[MATMUL_RESULT]] {{\[}}[0, 1, 2], [3]] : tensor<20x7xf32> into tensor<1x4x5x7xf32>
// CHECK: return %[[RESULT]]

// -----

// CHECK: @conv_2d_1x1_dyn
func.func @conv_2d_1x1_dyn(%input: tensor<1x4x?x2xf32>, %filter: tensor<1x1x2x7xf32>) -> tensor<1x4x?x7xf32> {
    %c2 = arith.constant 2 : index
    %d2 = tensor.dim %input, %c2 : tensor<1x4x?x2xf32>
    %0 = linalg.init_tensor [1, 4, %d2, 7] : tensor<1x4x?x7xf32>
    %1 = linalg.conv_2d_nhwc_hwcf {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x4x?x2xf32>, tensor<1x1x2x7xf32>) outs(%0 : tensor<1x4x?x7xf32>) -> tensor<1x4x?x7xf32>
    return %1 : tensor<1x4x?x7xf32>
}

// CHECK: %[[INPUT:.+]]: tensor<1x4x?x2xf32>
// CHECK: %[[FILTER:.+]]: tensor<1x1x2x7xf32>
// CHECK: %[[C2:.+]] = arith.constant 2 : index
// CHECK: %[[D2:.+]] = tensor.dim %[[INPUT]], %[[C2]]
// CHECK: %[[OTUPUT:.+]] = linalg.init_tensor [1, 4, %[[D2]], 7] : tensor<1x4x?x7xf32>
// CHECK: %[[RESHAPED_INPUT:.+]] = tensor.collapse_shape %[[INPUT]] {{\[}}[0, 1, 2], [3]] : tensor<1x4x?x2xf32> into tensor<?x2xf32>
// CHECK: %[[RESHAPED_FILTER:.+]] = tensor.collapse_shape %[[FILTER]] {{\[}}[0, 1, 2], [3]] : tensor<1x1x2x7xf32> into tensor<2x7xf32>
// CHECK: %[[RESHAPED_OUTPUT:.+]] = tensor.collapse_shape %[[OTUPUT]] {{\[}}[0, 1, 2], [3]] : tensor<1x4x?x7xf32> into tensor<?x7xf32>
// CHECK: %[[MATMUL_RESULT:.+]] = linalg.matmul ins(%[[RESHAPED_INPUT]], %[[RESHAPED_FILTER]] : tensor<?x2xf32>, tensor<2x7xf32>) outs(%[[RESHAPED_OUTPUT]] : tensor<?x7xf32>)
// CHECK: %[[RESULT:.+]] = tensor.expand_shape %[[MATMUL_RESULT]] {{\[}}[0, 1, 2], [3]] : tensor<?x7xf32> into tensor<1x4x?x7xf32>

// -----

func.func @conv_2d_1x1_fail_dyn(%input: tensor<1x?x?x2xf32>, %filter: tensor<1x1x2x7xf32>) -> tensor<1x?x?x7xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d1 = tensor.dim %input, %c1 : tensor<1x?x?x2xf32>
    %d2 = tensor.dim %input, %c2 : tensor<1x?x?x2xf32>
    %0 = linalg.init_tensor [1, %d1, %d2, 7] : tensor<1x?x?x7xf32>
    %1 = linalg.conv_2d_nhwc_hwcf {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x?x?x2xf32>, tensor<1x1x2x7xf32>) outs(%0 : tensor<1x?x?x7xf32>) -> tensor<1x?x?x7xf32>
    return %1 : tensor<1x?x?x7xf32>
}

// CHECK: @conv_2d_1x1_fail_dyn
// CHECK: linalg.conv_2d_nhwc_hwcf
