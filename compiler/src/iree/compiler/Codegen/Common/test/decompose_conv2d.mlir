// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-convolution-to-lower-dim-ops))" --split-input-file %s | FileCheck %s
// Test the same patterns on generic convolution ops by first generalizing the
// named ops. This ensures decomposition works on both named and generic convs.
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(linalg-generalize-named-ops,iree-codegen-decompose-convolution-to-lower-dim-ops))" --split-input-file %s | FileCheck %s --check-prefix=GENERIC

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0, 0, 0, 0], [1, 1, 1, 4, 0, 0], [0, 0, 0, 0, 1, 4], [0, 0, 0, 0, 0, 0]]>
module {
  func.func @restrict_num_workgroups(%input: tensor<1x1x4x4xf32>, %filter: tensor<1x4x4xf32>) -> tensor<1x1x1x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<1x1x1x4xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
    %conv = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, lowering_config = #config,
            strides = dense<1> : tensor<2xi64>} ins(%input, %filter : tensor<1x1x4x4xf32>, tensor<1x4x4xf32>) outs(%fill : tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
    return %conv : tensor<1x1x1x4xf32>
  }
}

// Verify that depthwise_conv_2d is decomposed to depthwise_conv_1d with updated tile sizes.
// The height dimension is collapsed, reducing tile sizes from 6D to 4D.

// CHECK: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 0, 0], [1, 1, 4, 0], [0, 0, 0, 4], [0, 0, 0, 0]]>
// CHECK-LABEL: func.func @restrict_num_workgroups
// CHECK-SAME:    %[[INPUT:.+]]: tensor<1x1x4x4xf32>
// CHECK-SAME:    %[[FILTER:.+]]: tensor<1x4x4xf32>
// CHECK-DAG:   %[[INPUT_SLICE:.+]] = tensor.extract_slice %[[INPUT]]{{.+}} : tensor<1x1x4x4xf32> to tensor<1x4x4xf32>
// CHECK-DAG:   %[[FILTER_SLICE:.+]] = tensor.extract_slice %[[FILTER]]{{.+}} : tensor<1x4x4xf32> to tensor<4x4xf32>
// CHECK:       linalg.depthwise_conv_1d_nwc_wc
// CHECK-SAME:    ins(%[[INPUT_SLICE]], %[[FILTER_SLICE]] : tensor<1x4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:    outs({{.+}} : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>

// For the generalized path: verify that the generic 2D conv is decomposed to 1D conv.
// GENERIC-LABEL: func.func @restrict_num_workgroups
// GENERIC-DAG:   %[[INPUT_SLICE:.+]] = tensor.extract_slice {{.+}} : tensor<1x1x4x4xf32> to tensor<1x4x4xf32>
// GENERIC-DAG:   %[[FILTER_SLICE:.+]] = tensor.extract_slice {{.+}} : tensor<1x4x4xf32> to tensor<4x4xf32>
// GENERIC:       linalg.depthwise_conv_1d_nwc_wc
// GENERIC-SAME:    ins(%[[INPUT_SLICE]], %[[FILTER_SLICE]] : tensor<1x4x4xf32>, tensor<4x4xf32>)
// GENERIC-SAME:    outs({{.+}} : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>

// -----

// Test case where output H > 1: should NOT decompose to 1D conv.
module {
  func.func @no_decompose_output_h_not_1(%input: tensor<1x4x4x4xf32>, %filter: tensor<1x4x4xf32>) -> tensor<1x4x1x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<1x4x1x4xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x4x1x4xf32>) -> tensor<1x4x1x4xf32>
    %conv = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>,
            strides = dense<1> : tensor<2xi64>} ins(%input, %filter : tensor<1x4x4x4xf32>, tensor<1x4x4xf32>) outs(%fill : tensor<1x4x1x4xf32>) -> tensor<1x4x1x4xf32>
    return %conv : tensor<1x4x1x4xf32>
  }
}

// CHECK-LABEL: func.func @no_decompose_output_h_not_1
// CHECK:       linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-NOT:   linalg.depthwise_conv_1d_nwc_wc

// GENERIC-LABEL: func.func @no_decompose_output_h_not_1
// GENERIC:       linalg.generic
// GENERIC-NOT:   linalg.depthwise_conv_1d_nwc_wc

// -----

// Test case where kernel H > 1: should NOT decompose to 1D conv.
module {
  func.func @no_decompose_kernel_h_not_1(%input: tensor<1x4x4x4xf32>, %filter: tensor<2x4x4xf32>) -> tensor<1x1x1x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<1x1x1x4xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
    %conv = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>,
            strides = dense<1> : tensor<2xi64>} ins(%input, %filter : tensor<1x4x4x4xf32>, tensor<2x4x4xf32>) outs(%fill : tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
    return %conv : tensor<1x1x1x4xf32>
  }
}

// CHECK-LABEL: func.func @no_decompose_kernel_h_not_1
// CHECK:       linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-NOT:   linalg.depthwise_conv_1d_nwc_wc

// GENERIC-LABEL: func.func @no_decompose_kernel_h_not_1
// GENERIC:       linalg.generic
// GENERIC-NOT:   linalg.depthwise_conv_1d_nwc_wc

// -----

// Test decomposition without lowering config.
module {
  func.func @decompose_without_config(%input: tensor<1x1x4x4xf32>, %filter: tensor<1x4x4xf32>) -> tensor<1x1x1x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<1x1x1x4xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
    %conv = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>,
            strides = dense<1> : tensor<2xi64>} ins(%input, %filter : tensor<1x1x4x4xf32>, tensor<1x4x4xf32>) outs(%fill : tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
    return %conv : tensor<1x1x1x4xf32>
  }
}

// Verify decomposition works even without lowering config.
// CHECK-LABEL: func.func @decompose_without_config
// CHECK:       linalg.depthwise_conv_1d_nwc_wc
// CHECK-NOT:   lowering_config

// GENERIC-LABEL: func.func @decompose_without_config
// GENERIC:       linalg.depthwise_conv_1d_nwc_wc
