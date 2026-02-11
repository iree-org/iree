// RUN: iree-opt --split-input-file --mlir-print-local-scope -iree-global-opt-convert-1x1-filter-conv2d-to-matmul %s | FileCheck %s

util.func public @nhwc_conv_2d(%input: tensor<1x4x5x2xf32>, %filter: tensor<1x1x2x7xf32>) -> tensor<1x4x5x7xf32> {
    %0 = tensor.empty() : tensor<1x4x5x7xf32>
    %1 = linalg.conv_2d_nhwc_hwcf {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x4x5x2xf32>, tensor<1x1x2x7xf32>) outs(%0 : tensor<1x4x5x7xf32>) -> tensor<1x4x5x7xf32>
    util.return %1 : tensor<1x4x5x7xf32>
}

// CHECK-LABEL: @nhwc_conv_2d
//       CHECK:   %[[RESULT:.*]] = linalg.generic
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d6)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
//       CHECK:   util.return %[[RESULT]]

// -----

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

// CHECK-LABEL: @dynamic_nhwc_conv_2d
//       CHECK:   %[[RESULT:.*]] = linalg.generic
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d6)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @nchw_conv_2d(%input: tensor<1x2x4x5xf32>, %filter: tensor<7x2x1x1xf32>) -> tensor<1x7x4x5xf32> {
    %0 = tensor.empty() : tensor<1x7x4x5xf32>
    %1 = linalg.conv_2d_nchw_fchw {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x2x4x5xf32>, tensor<7x2x1x1xf32>) outs(%0 : tensor<1x7x4x5xf32>) -> tensor<1x7x4x5xf32>
    util.return %1 : tensor<1x7x4x5xf32>
}
// CHECK-LABEL: @nchw_conv_2d
//       CHECK:   %[[RESULT:.*]] = linalg.generic
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
//       CHECK:   util.return %[[RESULT]]

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

// CHECK-LABEL: @dynamic_nchw_conv_2d
//       CHECK:   %[[RESULT:.*]] = linalg.generic
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @strided_nchw_conv_2d(%input: tensor<1x2x?x?xf32>, %filter: tensor<7x2x1x1xf32>, %d2 : index, %d3 : index) -> tensor<1x7x?x?xf32> {
    %0 = tensor.empty(%d2, %d3) : tensor<1x7x?x?xf32>
    %1 = linalg.conv_2d_nchw_fchw {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<2> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x2x?x?xf32>, tensor<7x2x1x1xf32>) outs(%0 : tensor<1x7x?x?xf32>) -> tensor<1x7x?x?xf32>
    util.return %1 : tensor<1x7x?x?xf32>
}

// CHECK-LABEL: @strided_nchw_conv_2d
//       CHECK:   %[[RESULT:.*]] = linalg.generic
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 * 2, d3 * 2)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @nhwc_fhwc_conv_2d(%input: tensor<1x4x5x2xf32>, %filter: tensor<7x1x1x2xf32>) -> tensor<1x4x5x7xf32> {
    %0 = tensor.empty() : tensor<1x4x5x7xf32>
    %1 = linalg.conv_2d_nhwc_fhwc {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x4x5x2xf32>, tensor<7x1x1x2xf32>) outs(%0 : tensor<1x4x5x7xf32>) -> tensor<1x4x5x7xf32>
    util.return %1 : tensor<1x4x5x7xf32>
}

// CHECK-LABEL: @nhwc_fhwc_conv_2d
//  CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]+]]: tensor<1x4x5x2xf32>
//  CHECK-SAME:   %[[FILTER:[a-zA-Z0-9_]+]]: tensor<7x1x1x2xf32>
//       CHECK:   %[[EMPTY_TRANSPOSE:.*]] = tensor.empty() : tensor<2x1x1x7xf32>
//       CHECK:   %[[TRANSPOSED_FILTER:.*]] = linalg.transpose ins(%[[FILTER]] : tensor<7x1x1x2xf32>) outs(%[[EMPTY_TRANSPOSE]] : tensor<2x1x1x7xf32>) permutation = [3, 1, 2, 0]
//       CHECK:   %[[RESULT:.*]] = linalg.generic
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d6)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6, d4, d5, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
//  CHECK-SAME:     ins(%[[INPUT]], %[[TRANSPOSED_FILTER]] :
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @dynamic_nhwc_fhwc_conv_2d(%input: tensor<1x4x?x2xf32>, %filter: tensor<7x1x1x2xf32>) -> tensor<1x4x?x7xf32> {
    %c2 = arith.constant 2 : index
    %d2 = tensor.dim %input, %c2 : tensor<1x4x?x2xf32>
    %0 = tensor.empty(%d2) : tensor<1x4x?x7xf32>
    %1 = linalg.conv_2d_nhwc_fhwc {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x4x?x2xf32>, tensor<7x1x1x2xf32>) outs(%0 : tensor<1x4x?x7xf32>) -> tensor<1x4x?x7xf32>
    util.return %1 : tensor<1x4x?x7xf32>
}

// CHECK-LABEL: @dynamic_nhwc_fhwc_conv_2d
//  CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]+]]: tensor<1x4x?x2xf32>
//  CHECK-SAME:   %[[FILTER:[a-zA-Z0-9_]+]]: tensor<7x1x1x2xf32>
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[DIM:.*]] = tensor.dim %[[INPUT]], %[[C2]]
//   CHECK-DAG:   %[[OUTPUT:.*]] = tensor.empty(%[[DIM]]) : tensor<1x4x?x7xf32>
//       CHECK:   %[[EMPTY_TRANSPOSE:.*]] = tensor.empty() : tensor<2x1x1x7xf32>
//       CHECK:   %[[TRANSPOSED_FILTER:.*]] = linalg.transpose ins(%[[FILTER]] : tensor<7x1x1x2xf32>) outs(%[[EMPTY_TRANSPOSE]] : tensor<2x1x1x7xf32>) permutation = [3, 1, 2, 0]
//       CHECK:   %[[RESULT:.*]] = linalg.generic
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d6)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6, d4, d5, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
//  CHECK-SAME:     ins(%[[INPUT]], %[[TRANSPOSED_FILTER]] :
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @strided_nhwc_fhwc_conv_2d(%input: tensor<1x?x?x2xf32>, %filter: tensor<7x1x1x2xf32>, %d1 : index, %d2 : index) -> tensor<1x?x?x7xf32> {
    %0 = tensor.empty(%d1, %d2) : tensor<1x?x?x7xf32>
    %1 = linalg.conv_2d_nhwc_fhwc {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<2> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x?x?x2xf32>, tensor<7x1x1x2xf32>) outs(%0 : tensor<1x?x?x7xf32>) -> tensor<1x?x?x7xf32>
    util.return %1 : tensor<1x?x?x7xf32>
}

// CHECK-LABEL: @strided_nhwc_fhwc_conv_2d
//  CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]+]]: tensor<1x?x?x2xf32>
//  CHECK-SAME:   %[[FILTER:[a-zA-Z0-9_]+]]: tensor<7x1x1x2xf32>
//       CHECK:   %[[EMPTY_TRANSPOSE:.*]] = tensor.empty() : tensor<2x1x1x7xf32>
//       CHECK:   %[[TRANSPOSED_FILTER:.*]] = linalg.transpose ins(%[[FILTER]] : tensor<7x1x1x2xf32>) outs(%[[EMPTY_TRANSPOSE]] : tensor<2x1x1x7xf32>) permutation = [3, 1, 2, 0]
//       CHECK:   %[[RESULT:.*]] = linalg.generic
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2, d2 * 2, d6)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6, d4, d5, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
//  CHECK-SAME:     ins(%[[INPUT]], %[[TRANSPOSED_FILTER]] :
//       CHECK:   util.return %[[RESULT]]
