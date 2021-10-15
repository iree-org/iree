func @reduce_min() {
  %input = flow.tensor.constant
    dense<[[1.0, 2.0, 3.0, 4.0],[-1.0 ,-2.0 ,-3.0 ,-4.0]]> : tensor<2x4xf32> -> tensor<?x?xf32>
  %0 = mhlo.constant dense<0x7F800000> : tensor<f32>
  %1 = "mhlo.reduce"(%input, %0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = mhlo.minimum %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%1, dense<-4.0> : tensor<f32>) : tensor<f32>
  return
}
