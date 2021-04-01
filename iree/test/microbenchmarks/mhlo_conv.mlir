// Naming convention @conv_{input-spatial-dim}_{output-spatial-dim}_{filter-size}-{outputsizexinputsize}

// The following ops sampled from MobileVision V1 (mobilenet_v1_100_224)
// https://github.com/google/iree/blob/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py#L34

func @conv_244_112_3x3_3x32() -> tensor<1x112x112x32xf32> attributes { iree.module.export } {
    %input = iree.unfoldable_constant dense<1.0> : tensor<1x224x224x3xf32>
    %filter = iree.unfoldable_constant dense<1.0> : tensor<3x3x3x32xf32>
    %0 = "mhlo.convolution"(%input, %filter) {
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
        rhs_dilation = dense<1> : tensor<2xi64>,
        window_strides = dense<2> : tensor<2xi64>
    } : (tensor<1x224x224x3xf32>, tensor<3x3x3x32xf32>) -> tensor<1x112x112x32xf32>
    return %0 : tensor<1x112x112x32xf32>
}

func @conv_112_112_1x1_32x64() -> tensor<1x112x112x64xf32> attributes { iree.module.export } {
    %input = iree.unfoldable_constant dense<1.0> : tensor<1x112x112x32xf32>
    %filter = iree.unfoldable_constant dense<1.0> : tensor<1x1x32x64xf32>
    %0 = "mhlo.convolution"(%input, %filter) {
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
        padding = dense<0> : tensor<2x2xi64>,
        rhs_dilation = dense<1> : tensor<2xi64>,
        window_strides = dense<1> : tensor<2xi64>
    } : (tensor<1x112x112x32xf32>, tensor<1x1x32x64xf32>) -> tensor<1x112x112x64xf32>
    return %0 : tensor<1x112x112x64xf32>
}

func @conv_7_7_1x1_1024x1024() -> tensor<1x7x7x1024xf32> attributes { iree.module.export } {
    %input = iree.unfoldable_constant dense<1.0> : tensor<1x7x7x1024xf32>
    %filter = iree.unfoldable_constant dense<1.0> : tensor<1x1x1024x1024xf32>
    %0 = "mhlo.convolution"(%input, %filter) {
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
        padding = dense<0> : tensor<2x2xi64>,
        rhs_dilation = dense<1> : tensor<2xi64>,
        window_strides = dense<1> : tensor<2xi64>
    } : (tensor<1x7x7x1024xf32>, tensor<1x1x1024x1024xf32>) -> tensor<1x7x7x1024xf32>
    return %0 : tensor<1x7x7x1024xf32>
}
