// RUN: iree-opt --split-input-file -iree-global-opt-convert-1x1-filter-conv2d-to-matmul %s | FileCheck %s

util.func public @nhwc_conv_2d(%input: tensor<1x4x5x2xf32>, %filter: tensor<1x1x2x7xf32>) -> tensor<1x4x5x7xf32> {
    %0 = tensor.empty() : tensor<1x4x5x7xf32>
    %1 = linalg.conv_2d_nhwc_hwcf {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x4x5x2xf32>, tensor<1x1x2x7xf32>) outs(%0 : tensor<1x4x5x7xf32>) -> tensor<1x4x5x7xf32>
    util.return %1 : tensor<1x4x5x7xf32>
}

// CHECK: @nhwc_conv_2d
// CHECK: %[[INPUT:.+]]: tensor<1x4x5x2xf32>
// CHECK: %[[FILTER:.+]]: tensor<1x1x2x7xf32>
// CHECK: %[[OUTPUT:.+]] = tensor.empty() : tensor<1x4x5x7xf32>
// CHECK: %[[RESHAPED_INPUT:.+]] = tensor.collapse_shape %[[INPUT]] {{\[}}[0, 1, 2], [3]] : tensor<1x4x5x2xf32> into tensor<20x2xf32>
// CHECK: %[[RESHAPED_FILTER:.+]] = tensor.collapse_shape %[[FILTER]] {{\[}}[0, 1, 2], [3]] : tensor<1x1x2x7xf32> into tensor<2x7xf32>
// CHECK: %[[RESHAPED_OUTPUT:.+]] = tensor.collapse_shape %[[OUTPUT]] {{\[}}[0, 1, 2], [3]] : tensor<1x4x5x7xf32> into tensor<20x7xf32>
// CHECK: %[[MATMUL_RESULT:.+]] = linalg.matmul ins(%[[RESHAPED_INPUT]], %[[RESHAPED_FILTER]] : tensor<20x2xf32>, tensor<2x7xf32>) outs(%[[RESHAPED_OUTPUT]] : tensor<20x7xf32>)
// CHECK: %[[RESULT:.+]] = tensor.expand_shape %[[MATMUL_RESULT]] {{\[}}[0, 1, 2], [3]] : tensor<20x7xf32> into tensor<1x4x5x7xf32>
// CHECK: util.return %[[RESULT]]

// -----

// CHECK: @dynamic_nhwc_conv_2d
util.func public @dynamic_nhwc_conv_2d(%input: tensor<1x4x?x2xf32>, %filter: tensor<1x1x2x7xf32>) -> tensor<1x4x?x7xf32> {
    %c2 = arith.constant 2 : index
    %d2 = tensor.dim %input, %c2 : tensor<1x4x?x2xf32>
    %0 = tensor.empty(%d2) : tensor<1x4x?x7xf32>
    %1 = linalg.conv_2d_nhwc_hwcf {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x4x?x2xf32>, tensor<1x1x2x7xf32>) outs(%0 : tensor<1x4x?x7xf32>) -> tensor<1x4x?x7xf32>
    util.return %1 : tensor<1x4x?x7xf32>
}

// CHECK: %[[INPUT:.+]]: tensor<1x4x?x2xf32>
// CHECK: %[[FILTER:.+]]: tensor<1x1x2x7xf32>
// CHECK: %[[C2:.+]] = arith.constant 2 : index
// CHECK: %[[D2:.+]] = tensor.dim %[[INPUT]], %[[C2]]
// CHECK: %[[OUTPUT:.+]] = tensor.empty(%[[D2]]) : tensor<1x4x?x7xf32>
// CHECK: %[[RESHAPED_INPUT:.+]] = tensor.collapse_shape %[[INPUT]] {{\[}}[0, 1, 2], [3]] : tensor<1x4x?x2xf32> into tensor<?x2xf32>
// CHECK: %[[RESHAPED_FILTER:.+]] = tensor.collapse_shape %[[FILTER]] {{\[}}[0, 1, 2], [3]] : tensor<1x1x2x7xf32> into tensor<2x7xf32>
// CHECK: %[[RESHAPED_OUTPUT:.+]] = tensor.collapse_shape %[[OUTPUT]] {{\[}}[0, 1, 2], [3]] : tensor<1x4x?x7xf32> into tensor<?x7xf32>
// CHECK: %[[MATMUL_RESULT:.+]] = linalg.matmul ins(%[[RESHAPED_INPUT]], %[[RESHAPED_FILTER]] : tensor<?x2xf32>, tensor<2x7xf32>) outs(%[[RESHAPED_OUTPUT]] : tensor<?x7xf32>)
// CHECK: %[[RESULT:.+]] = tensor.expand_shape %[[MATMUL_RESULT]] {{\[}}[0, 1, 2], [3]] : tensor<?x7xf32> into tensor<1x4x?x7xf32>

// -----

util.func public @fail_dynamic_nhwc_conv_2d(%input: tensor<1x?x?x2xf32>, %filter: tensor<1x1x2x7xf32>) -> tensor<1x?x?x7xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d1 = tensor.dim %input, %c1 : tensor<1x?x?x2xf32>
    %d2 = tensor.dim %input, %c2 : tensor<1x?x?x2xf32>
    %0 = tensor.empty(%d1, %d2) : tensor<1x?x?x7xf32>
    %1 = linalg.conv_2d_nhwc_hwcf {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x?x?x2xf32>, tensor<1x1x2x7xf32>) outs(%0 : tensor<1x?x?x7xf32>) -> tensor<1x?x?x7xf32>
    util.return %1 : tensor<1x?x?x7xf32>
}

// CHECK: @fail_dynamic_nhwc_conv_2d
// CHECK: linalg.conv_2d_nhwc_hwcf

// -----

util.func public @nchw_conv_2d(%input: tensor<1x2x4x5xf32>, %filter: tensor<7x2x1x1xf32>) -> tensor<1x7x4x5xf32> {
    %0 = tensor.empty() : tensor<1x7x4x5xf32>
    %1 = linalg.conv_2d_nchw_fchw {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x2x4x5xf32>, tensor<7x2x1x1xf32>) outs(%0 : tensor<1x7x4x5xf32>) -> tensor<1x7x4x5xf32>
    util.return %1 : tensor<1x7x4x5xf32>
}
// CHECK: @nchw_conv_2d
// CHECK: %[[INPUT:.+]]: tensor<1x2x4x5xf32>
// CHECK: %[[FILTER:.+]]: tensor<7x2x1x1xf32>
// CHECK: %[[OUTPUT:.+]] = tensor.empty() : tensor<1x7x4x5xf32>
// CHECK: %[[RESHAPED_INPUT:.+]] = tensor.collapse_shape %[[INPUT]] {{\[}}[0, 1], [2, 3]] : tensor<1x2x4x5xf32> into tensor<2x20xf32>
// CHECK: %[[RESHAPED_FILTER:.+]] = tensor.collapse_shape %[[FILTER]] {{\[}}[0], [1, 2, 3]] : tensor<7x2x1x1xf32> into tensor<7x2xf32>
// CHECK: %[[RESHAPED_OUTPUT:.+]] = tensor.collapse_shape %[[OUTPUT]] {{\[}}[0, 1], [2, 3]] : tensor<1x7x4x5xf32> into tensor<7x20xf32>
// CHECK: %[[MATMUL_RESULT:.+]] = linalg.matmul ins(%[[RESHAPED_FILTER]], %[[RESHAPED_INPUT]] : tensor<7x2xf32>, tensor<2x20xf32>) outs(%[[RESHAPED_OUTPUT]] : tensor<7x20xf32>)
// CHECK: %[[RESULT:.+]] = tensor.expand_shape %[[MATMUL_RESULT]] {{\[}}[0, 1], [2, 3]] : tensor<7x20xf32> into tensor<1x7x4x5xf32>
// CHECK: util.return %[[RESULT]]

// -----

util.func public @dynamic_nchw_conv_2d(%input: tensor<1x2x4x?xf32>, %filter: tensor<7x2x1x1xf32>) -> tensor<1x7x4x?xf32> {
    %c3 = arith.constant 3 : index
    %d3 = tensor.dim %input, %c3 : tensor<1x2x4x?xf32>
    %0 = tensor.empty(%d3) : tensor<1x7x4x?xf32>
    %1 = linalg.conv_2d_nchw_fchw {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x2x4x?xf32>, tensor<7x2x1x1xf32>) outs(%0 : tensor<1x7x4x?xf32>) -> tensor<1x7x4x?xf32>
    util.return %1 : tensor<1x7x4x?xf32>
}

// CHECK: @dynamic_nchw_conv_2d
// CHECK: %[[INPUT:.+]]: tensor<1x2x4x?xf32>
// CHECK: %[[FILTER:.+]]: tensor<7x2x1x1xf32>
// CHECK: %[[C3:.+]] = arith.constant 3 : index
// CHECK: %[[D3:.+]] = tensor.dim %[[INPUT]], %[[C3]]
// CHECK: %[[OUTPUT:.+]] = tensor.empty(%[[D3]]) : tensor<1x7x4x?xf32>
// CHECK: %[[RESHAPED_INPUT:.+]] = tensor.collapse_shape %[[INPUT]] {{\[}}[0, 1], [2, 3]] : tensor<1x2x4x?xf32> into tensor<2x?xf32>
// CHECK: %[[RESHAPED_FILTER:.+]] = tensor.collapse_shape %[[FILTER]] {{\[}}[0], [1, 2, 3]] : tensor<7x2x1x1xf32> into tensor<7x2xf32>
// CHECK: %[[RESHAPED_OUTPUT:.+]] = tensor.collapse_shape %[[OUTPUT]] {{\[}}[0, 1], [2, 3]] : tensor<1x7x4x?xf32> into tensor<7x?xf32>
// CHECK: %[[MATMUL_RESULT:.+]] = linalg.matmul ins(%[[RESHAPED_FILTER]], %[[RESHAPED_INPUT]] : tensor<7x2xf32>, tensor<2x?xf32>) outs(%[[RESHAPED_OUTPUT]] : tensor<7x?xf32>)
// CHECK: %[[RESULT:.+]] = tensor.expand_shape %[[MATMUL_RESULT]] {{\[}}[0, 1], [2, 3]] : tensor<7x?xf32> into tensor<1x7x4x?xf32>
// CHECK: util.return %[[RESULT]]

// -----

util.func public @fail_dynamic_nchw_conv_2d(%input: tensor<1x2x?x?xf32>, %filter: tensor<7x2x1x1xf32>) -> tensor<1x7x?x?xf32> {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %d2 = tensor.dim %input, %c2 : tensor<1x2x?x?xf32>
    %d3 = tensor.dim %input, %c3 : tensor<1x2x?x?xf32>
    %0 = tensor.empty(%d2, %d3) : tensor<1x7x?x?xf32>
    %1 = linalg.conv_2d_nchw_fchw {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x2x?x?xf32>, tensor<7x2x1x1xf32>) outs(%0 : tensor<1x7x?x?xf32>) -> tensor<1x7x?x?xf32>
    util.return %1 : tensor<1x7x?x?xf32>
}

// CHECK: @fail_dynamic_nchw_conv_2d
// CHECK: linalg.conv_2d_nchw_fchw
