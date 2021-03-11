// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-tensors %s | IreeFileCheck %s

module {
  func @depthwise_conv() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x4x5x2xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<2x2x2x3xf32>
    %2 = "mhlo.convolution"(%0, %1) {
      batch_group_count = 1 : i64,
      dimension_numbers = {
        input_batch_dimension = 0 : i64,
        input_feature_dimension = 3 : i64,
        input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
        kernel_input_feature_dimension = 2 : i64,
        kernel_output_feature_dimension = 3 : i64,
        kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
        output_batch_dimension = 0 : i64,
        output_feature_dimension = 3 : i64,
        output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>
      },
      feature_group_count = 2 : i64,
      padding = dense<0> : tensor<2x2xi64>,
      rhs_dilation = dense<1> : tensor<2xi64>,
      window_strides = dense<1> : tensor<2xi64>} : (tensor<2x4x5x2xf32>, tensor<2x2x2x3xf32>) -> tensor<2x3x4x6xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<2x3x4x6xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
// CHECK:     func @depthwise_conv()
// CHECK-DAG:   %[[IN:.+]] = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x4x5x2xf32>
// CHECK-DAG:   %[[FILTER:.+]] = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<2x2x2x3xf32>
// CHECK:       %[[INIT:.+]] = linalg.init_tensor [2, 3, 4, 2, 3] : tensor<2x3x4x2x3xf32>
// CHECK:       %[[CST:.+]] = constant 0.000000e+00 : f32
// CHECK:       %[[FILL:.+]] = linalg.fill(%[[INIT]], %[[CST]]) : tensor<2x3x4x2x3xf32>, f32 -> tensor<2x3x4x2x3xf32>
// CHECK:       %[[OUT:.+]] = linalg.depthwise_conv_2d_input_nhwc_filter_hwcf
// CHECK-SAME:    {strides = dense<1> : tensor<2xi64>}
// CHECK-SAME:     ins(%[[IN]], %[[FILTER]] : tensor<2x4x5x2xf32>, tensor<2x2x2x3xf32>)
// CHECK-SAME:    outs(%[[FILL]] : tensor<2x3x4x2x3xf32>) -> tensor<2x3x4x2x3xf32>
// CHECK:       %{{.+}} = linalg.tensor_reshape %[[OUT]]
// CHECK-SAME:    [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]]]
// CHECK-SAME:    : tensor<2x3x4x2x3xf32> into tensor<2x3x4x6xf32>

// -----

module {
  func @depthwise_conv_multiplier_1() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<1x113x113x96xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<3x3x1x96xf32>
    %2 = "mhlo.convolution"(%0, %1) {
      batch_group_count = 1 : i64,
      dimension_numbers = {
        input_batch_dimension = 0 : i64,
        input_feature_dimension = 3 : i64,
        input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
        kernel_input_feature_dimension = 2 : i64,
        kernel_output_feature_dimension = 3 : i64,
        kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
        output_batch_dimension = 0 : i64,
        output_feature_dimension = 3 : i64,
        output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>
      },
      feature_group_count = 96 : i64,
      padding = dense<0> : tensor<2x2xi64>,
      rhs_dilation = dense<1> : tensor<2xi64>,
      window_strides = dense<2> : tensor<2xi64>} : (tensor<1x113x113x96xf32>, tensor<3x3x1x96xf32>) -> tensor<1x56x56x96xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<1x56x56x96xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
// CHECK:     func @depthwise_conv_multiplier_1()
// CHECK-DAG:   %[[IN:.+]] = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<1x113x113x96xf32>
// CHECK-DAG:   %[[FILTER:.+]] = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<3x3x1x96xf32>
// CHECK:       %[[INIT:.+]] = linalg.init_tensor [1, 56, 56, 96] : tensor<1x56x56x96xf32>
// CHECK:       %[[CST:.+]] = constant 0.000000e+00 : f32
// CHECK:       %[[FILL:.+]] = linalg.fill(%[[INIT]], %[[CST]]) : tensor<1x56x56x96xf32>, f32 -> tensor<1x56x56x96xf32>
// CHECK:       %[[RESHAPED_FILTER:.+]] = linalg.tensor_reshape %[[FILTER]] [#[[MAP0]], #[[MAP1]], #[[MAP2]]] : tensor<3x3x1x96xf32> into tensor<3x3x96xf32>
// CHECK:       %{{.+}} = linalg.depthwise_conv_2d_input_nhwc_filter_hwc
// CHECK-SAME:    {strides = dense<2> : tensor<2xi64>}
// CHECK-SAME:     ins(%[[IN]], %[[RESHAPED_FILTER]] : tensor<1x113x113x96xf32>, tensor<3x3x96xf32>)
// CHECK-SAME:    outs(%[[FILL]] : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
