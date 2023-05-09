func.func @batchnorm_inference_4x2() {
  %x = util.unfoldable_constant dense<[[1.0, 2.0, 3.0, 4.0],[5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
  %mean = util.unfoldable_constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
  %var = util.unfoldable_constant dense<[2.0, 2.0, 2.0, 2.0]> : tensor<4xf32>
  %offset = util.unfoldable_constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
  %scale = util.unfoldable_constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
  %result = "stablehlo.batch_norm_inference"(%x, %mean, %var, %offset, %scale) {epsilon = 1.000000e-03 : f32, feature_index = 1 : i64} : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  // TODO(gcmn): This should probably be a fuzzier check with round values.
  check.expect_almost_eq_const(%result, dense<[
      [2.0, 2.9995, 3.999, 4.9985],
      [5.998, 6.9975, 7.997, 8.9965]]> : tensor<2x4xf32>) : tensor<2x4xf32>
  return
}
