func.func @scalar() {
  %input1 = util.unfoldable_constant dense<16.0> : tensor<f32>
  %input2 = util.unfoldable_constant dense<7.0> : tensor<f32>
  %result = "mhlo.remainder"(%input1, %input2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<2.0> : tensor<f32>) : tensor<f32>
  return
}

func.func @tensor() {
  %input1 = util.unfoldable_constant dense<[16.0, 17.0, 18.0]> : tensor<3xf32>
  %input2 = util.unfoldable_constant dense<[7.0, 8.0, 9.0]> : tensor<3xf32>
  %result = "mhlo.remainder"(%input1, %input2) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  check.expect_almost_eq_const(%result, dense<[2.0, 1.0, 0.0]> : tensor<3xf32>) : tensor<3xf32>
  return
}

func.func @negative_den() {
  %input1 = util.unfoldable_constant dense<16.0> : tensor<f32>
  %input2 = util.unfoldable_constant dense<-7.0> : tensor<f32>
  %result = "mhlo.remainder"(%input1, %input2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<2.0> : tensor<f32>) : tensor<f32>
  return
}

func.func @negative_num() {
  %input1 = util.unfoldable_constant dense<-16.0> : tensor<f32>
  %input2 = util.unfoldable_constant dense<7.0> : tensor<f32>
  %result = "mhlo.remainder"(%input1, %input2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<-2.0> : tensor<f32>) : tensor<f32>
  return
}

func.func @positive_self() {
  %input = util.unfoldable_constant dense<[1., 2., 3., 32., 33., 34., 40., 41., 42., 46., 47., 48., 108., 109., 110., 111.]> : tensor<16xf32>
  %result = "mhlo.remainder"(%input, %input) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  check.expect_almost_eq_const(%result, dense<0.0> : tensor<16xf32>) : tensor<16xf32>
  return
}
func.func @negative_self() {
  %input = util.unfoldable_constant dense<[-1., -2., -3., -32., -33., -34., -40., -41., -42., -46., -47., -48., -108., -109., -110., -111.]> : tensor<16xf32>
  %result = "mhlo.remainder"(%input, %input) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  check.expect_almost_eq_const(%result, dense<0.0> : tensor<16xf32>) : tensor<16xf32>
  return
}

func.func @scalar_int() {
  %input1 = util.unfoldable_constant dense<16> : tensor<i32>
  %input2 = util.unfoldable_constant dense<7> : tensor<i32>
  %result = "mhlo.remainder"(%input1, %input2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%result, dense<2> : tensor<i32>) : tensor<i32>
  return
}

func.func @tensor_int() {
  %input1 = util.unfoldable_constant dense<[16, 17, 18]> : tensor<3xi32>
  %input2 = util.unfoldable_constant dense<[7, 8, 9]> : tensor<3xi32>
  %result = "mhlo.remainder"(%input1, %input2) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  check.expect_eq_const(%result, dense<[2, 1, 0]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func.func @negative_den_int() {
  %input1 = util.unfoldable_constant dense<16> : tensor<i32>
  %input2 = util.unfoldable_constant dense<-7> : tensor<i32>
  %result = "mhlo.remainder"(%input1, %input2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%result, dense<2> : tensor<i32>) : tensor<i32>
  return
}

func.func @negative_num_int() {
  %input1 = util.unfoldable_constant dense<-16> : tensor<i32>
  %input2 = util.unfoldable_constant dense<7> : tensor<i32>
  %result = "mhlo.remainder"(%input1, %input2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%result, dense<-2> : tensor<i32>) : tensor<i32>
  return
}
