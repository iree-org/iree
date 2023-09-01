func.func @tensor_i32() {
  %lhs = util.unfoldable_constant dense<[1, 2, 7, 4]> : tensor<4xi32>
  %rhs = util.unfoldable_constant dense<[5, 2, 3, 4]> : tensor<4xi32>
  %result = "stablehlo.minimum"(%lhs, %rhs) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[1, 2, 3, 4]> : tensor<4xi32>) : tensor<4xi32>
  return
}

func.func @tensor_odd_dim() {
  %lhs = util.unfoldable_constant dense<[1, 2, 7]> : tensor<3xi32>
  %rhs = util.unfoldable_constant dense<[5, 2, 3]> : tensor<3xi32>
  %result = "stablehlo.minimum"(%lhs, %rhs) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  check.expect_eq_const(%result, dense<[1, 2, 3]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func.func @scalar_i32() {
  %lhs = util.unfoldable_constant dense<1> : tensor<i32>
  %rhs = util.unfoldable_constant dense<2> : tensor<i32>
  %result = "stablehlo.minimum"(%lhs, %rhs) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%result, dense<1> : tensor<i32>) : tensor<i32>
  return
}

func.func @negative_i32() {
  %lhs = util.unfoldable_constant dense<1> : tensor<i32>
  %rhs = util.unfoldable_constant dense<-2> : tensor<i32>
  %result = "stablehlo.minimum"(%lhs, %rhs) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%result, dense<-2> : tensor<i32>) : tensor<i32>
  return
}

func.func @i8() {
  %lhs = util.unfoldable_constant dense<1> : tensor<i8>
  %rhs = util.unfoldable_constant dense<2> : tensor<i8>
  %result = "stablehlo.minimum"(%lhs, %rhs) : (tensor<i8>, tensor<i8>) -> tensor<i8>
  check.expect_eq_const(%result, dense<1> : tensor<i8>) : tensor<i8>
  return
}

func.func @i16() {
  %lhs = util.unfoldable_constant dense<1> : tensor<i16>
  %rhs = util.unfoldable_constant dense<2> : tensor<i16>
  %result = "stablehlo.minimum"(%lhs, %rhs) : (tensor<i16>, tensor<i16>) -> tensor<i16>
  check.expect_eq_const(%result, dense<1> : tensor<i16>) : tensor<i16>
  return
}

func.func @i64() {
  %lhs = util.unfoldable_constant dense<1> : tensor<i64>
  %rhs = util.unfoldable_constant dense<2> : tensor<i64>
  %result = "stablehlo.minimum"(%lhs, %rhs) : (tensor<i64>, tensor<i64>) -> tensor<i64>
  check.expect_eq_const(%result, dense<1> : tensor<i64>) : tensor<i64>
  return
}

func.func @tensor_f32() {
  %lhs = util.unfoldable_constant dense<[1.0, 2.0, 7.0, 4.0]> : tensor<4xf32>
  %rhs = util.unfoldable_constant dense<[5.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %result = "stablehlo.minimum"(%lhs, %rhs) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func.func @scalar_f32() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<f32>
  %rhs = util.unfoldable_constant dense<2.0> : tensor<f32>
  %result = "stablehlo.minimum"(%lhs, %rhs) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<1.0> : tensor<f32>) : tensor<f32>
  return
}

func.func @double() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<f64>
  %rhs = util.unfoldable_constant dense<2.0> : tensor<f64>
  %result = "stablehlo.minimum"(%lhs, %rhs) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  check.expect_almost_eq_const(%result, dense<1.0> : tensor<f64>) : tensor<f64>
  return
}

func.func @negative_f32() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<f32>
  %rhs = util.unfoldable_constant dense<-2.0> : tensor<f32>
  %result = "stablehlo.minimum"(%lhs, %rhs) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<-2.0> : tensor<f32>) : tensor<f32>
  return
}
