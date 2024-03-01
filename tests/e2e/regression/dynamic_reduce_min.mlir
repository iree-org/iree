func.func @reduce_min() {
  %input = flow.tensor.constant
    dense<[[1.0, 2.0, 3.0, 4.0],[-1.0 ,-2.0 ,-3.0 ,-4.0]]> : tensor<2x4xf32> -> tensor<?x?xf32>
  %0 = stablehlo.constant dense<0x7F800000> : tensor<f32>
  %1 = "stablehlo.reduce"(%input, %0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = stablehlo.minimum %arg1, %arg2 : tensor<f32>
    "stablehlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 0, 1>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%1, dense<-4.0> : tensor<f32>) : tensor<f32>
  return
}
