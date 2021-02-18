// RUN: iree-opt -split-input-file -iree-codegen-convert-1x1-conv-to-dot %s | IreeFileCheck %s

// CHECK: @conv_1x1(%[[INPUT:.+]]: tensor<2x4x5x2xf32>, %[[FILTER:.+]]: tensor<1x1x2x7xf32>) -> tensor<2x4x5x7xf32>
func @conv_1x1(%arg0: tensor<2x4x5x2xf32>, %arg1: tensor<1x1x2x7xf32>) -> tensor<2x4x5x7xf32> {
    // CHECK: %[[RESHAPED_INPUT:.+]] = "mhlo.reshape"(%[[INPUT]]) : (tensor<2x4x5x2xf32>) -> tensor<40x2xf32>
    // CHECK: %[[RESHAPED_FILTER:.+]] = "mhlo.reshape"(%[[FILTER]]) : (tensor<1x1x2x7xf32>) -> tensor<2x7xf32>
    // CHECK: %[[DOT_RESULT:.+]] = "mhlo.dot"(%[[RESHAPED_INPUT]], %[[RESHAPED_FILTER]]) {precision_config = ["HIGHEST", "HIGHEST"]} : (tensor<40x2xf32>, tensor<2x7xf32>) -> tensor<40x7xf32>
    // CEHCK: %[[RESULT:.+]] = "mhlo.reshape"(%[[DOT_RESULT]]) : (tensor<40x7xf32>) -> tensor<2x4x5x7xf32>
    %0 = "mhlo.convolution"(%arg0, %arg1) {
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
     padding = dense<0> : tensor<2x2xi64>,
     rhs_dilation = dense<1> : tensor<2xi64>,
     window_strides = dense<1> : tensor<2xi64>} : (tensor<2x4x5x2xf32>, tensor<1x1x2x7xf32>) -> tensor<2x4x5x7xf32>
    return %0 : tensor<2x4x5x7xf32>
}


