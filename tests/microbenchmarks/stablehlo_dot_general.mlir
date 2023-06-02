// The following ops are sampled from mobile_bert
// https://github.com/openxla/iree/blob/main/integrations/tensorflow/e2e/mobile_bert_squad_test.py

func.func @dot_general_4x384x32x384() -> tensor<4x384x384xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<4x384x32xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<4x32x384xf32>
    %0 = "stablehlo.dot_general"(%lhs, %rhs) {
        dot_dimension_numbers = #stablehlo.dot<
            lhs_batching_dimensions = [0],
            lhs_contracting_dimensions = [2],
            rhs_batching_dimensions = [0],
            rhs_contracting_dimensions = [1],
        >,
        precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<4x384x32xf32>, tensor<4x32x384xf32>) -> tensor<4x384x384xf32>
    return %0 : tensor<4x384x384xf32>
}

func.func @dot_general_4x384x384x32() -> tensor<4x384x32xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<4x384x384xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<4x384x32xf32>
    %0 = "stablehlo.dot_general"(%lhs, %rhs) {
        dot_dimension_numbers = #stablehlo.dot<
            lhs_batching_dimensions = [0],
            lhs_contracting_dimensions = [2],
            rhs_batching_dimensions = [0],
            rhs_contracting_dimensions = [1],
        >,
        precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<4x384x384xf32>, tensor<4x384x32xf32>) -> tensor<4x384x32xf32>
    return %0 : tensor<4x384x32xf32>
}
