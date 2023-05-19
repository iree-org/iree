func.func @multiple_results(
    %input_0 : tensor<f32>,
    %input_1 : tensor<f32>
) -> (tensor<f32>, tensor<f32>) {
  %result_0 = math.absf %input_0 : tensor<f32>
  %result_1 = math.absf %input_1 : tensor<f32>
  return %result_0, %result_1 : tensor<f32>, tensor<f32>
}
