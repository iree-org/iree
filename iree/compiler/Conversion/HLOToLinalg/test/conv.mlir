// RUN: iree-opt -iree-codegen-hlo-to-linalg-on-buffers %s | IreeFileCheck %s

module {
  // CHECK: func @conv
  func @conv() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<3x5x5x3xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<2x2x3x4xf32>
    //      CHECK: linalg.conv(%{{.+}}, %{{.+}}, %{{.+}}) {
    // CHECK-SAME: dilations = [1, 2]
    // CHECK-SAME: padding = dense<[
    // CHECK-SAME:                  [0, 1], [0, 1]]> : tensor<2x2xi64>
    // CHECK-SAME: strides = [2, 1]}
    %2 = "mhlo.convolution"(%1, %0) {
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
      feature_group_count = 1 : i64,
      padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>,
      rhs_dilation = dense<[1, 2]> : tensor<2xi64>,
      window_strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x2x3x4xf32>, tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<3x5x5x4xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
