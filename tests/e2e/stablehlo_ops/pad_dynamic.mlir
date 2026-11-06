// pad of dynamically shaped operands with negative edge padding.

func.func @pad_negative_and_interior() {
  %a = flow.tensor.dynamic_constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]> : tensor<3x4xf32> -> tensor<?x?xf32>
  %v = stablehlo.constant dense<-1.0> : tensor<f32>
  %r = stablehlo.pad %a, %v, low = [1, -1], high = [-2, 3], interior = [0, 1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  %s = tensor.cast %r : tensor<?x?xf32> to tensor<2x9xf32>
  check.expect_eq_const(%s, dense<[[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, 2.0, -1.0, 3.0, -1.0, 4.0, -1.0, -1.0, -1.0]]> : tensor<2x9xf32>) : tensor<2x9xf32>
  return
}

func.func @pad_negative_low_and_high() {
  %a = flow.tensor.dynamic_constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]> : tensor<3x4xf32> -> tensor<?x?xf32>
  %v = stablehlo.constant dense<-1.0> : tensor<f32>
  %r = stablehlo.pad %a, %v, low = [-1, 2], high = [1, -2], interior = [0, 0] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  %s = tensor.cast %r : tensor<?x?xf32> to tensor<3x4xf32>
  check.expect_eq_const(%s, dense<[[-1.0, -1.0, 5.0, 6.0], [-1.0, -1.0, 9.0, 10.0], [-1.0, -1.0, -1.0, -1.0]]> : tensor<3x4xf32>) : tensor<3x4xf32>
  return
}
