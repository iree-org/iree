// RUN: iree-run-mlir -iree-hal-target-backends=llvm-ir %s | IreeFileCheck %s

func @conv2d_nopadding() -> tensor<1x2x3x1xf32> attributes {iree.module.export} {
  %0 = iree.unfoldable_constant dense<[[
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0]],
        [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0], [29.0, 30.0]],
        [[31.0, 32.0], [33.0, 34.0], [35.0, 36.0], [37.0, 38.0], [39.0, 40.0]]]]> : tensor<1x4x5x2xf32>
  %1 = iree.unfoldable_constant dense<[[
        [[1.0], [2.0]], [[3.0], [4.0]]],
        [[[5.0], [6.0]], [[7.0], [8.0]]],
        [[[9.0], [10.0]], [[11.0], [12.0]]]]> : tensor<3x2x2x1xf32>
  %2 = "xla_hlo.conv"(%0, %1) {
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
          output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>},
        feature_group_count = 1 : i64,
        rhs_dilation = dense<1> : tensor<2xi64>,
        window_strides = dense<1> : tensor<2xi64>} : (tensor<1x4x5x2xf32>, tensor<3x2x2x1xf32>) -> tensor<1x2x3x1xf32>
  return %2 : tensor<1x2x3x1xf32>
}
// CHECK: 1x2x3x1xf32=[
// CHECK-SAME: [
// CHECK-SAME: [1310][1466][1622]
// CHECK-SAME: ][
// CHECK-SAME: [2090][2246][2402]
// CHECK-SAME: ]
// CHECK-SAME: ]
