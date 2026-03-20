// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-convolution-to-lower-dim-ops))" --split-input-file %s | FileCheck %s

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
