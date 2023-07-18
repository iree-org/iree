func.func @f32() {
  %lhs = util.unfoldable_constant dense<[
    [15.0, 14.0, 13.0],
    [12.0, 11.0, 10.0],
    [09.0, 08.0, 07.0],
    [06.0, 05.0, 04.0],
    [03.0, 02.0, 01.0]]> : tensor<5x3xf32>
  %rhs = util.unfoldable_constant dense<[
    [15.0, 14.0, 13.0, 12.0, 11.0],
    [10.0, 09.0, 08.0, 07.0, 06.0],
    [05.0, 04.0, 03.0, 02.0, 01.0]]> : tensor<3x5xf32>
  %res = "stablehlo.dot"(%lhs, %rhs) : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>
  check.expect_almost_eq_const(%res, dense<[
    [430.0, 388.0, 346.0, 304.0, 262.0],
    [340.0, 307.0, 274.0, 241.0, 208.0],
    [250.0, 226.0, 202.0, 178.0, 154.0],
    [160.0, 145.0, 130.0, 115.0, 100.0],
    [70.0, 64.0, 58.0, 52.0, 46.0]]> : tensor<5x5xf32>) : tensor<5x5xf32>
  return
}

func.func @i32i32.i32() {
  %lhs = util.unfoldable_constant dense<3> : tensor<2x4xi32>
  %rhs = util.unfoldable_constant dense<2> : tensor<4x2xi32>
  %res = "stablehlo.dot"(%lhs, %rhs) : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>
  check.expect_eq_const(%res, dense<24> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

func.func @i8i8.i32() {
  %lhs = util.unfoldable_constant dense<3> : tensor<2x4xi8>
  %rhs = util.unfoldable_constant dense<2> : tensor<4x2xi8>
  %res = "stablehlo.dot"(%lhs, %rhs) : (tensor<2x4xi8>, tensor<4x2xi8>) -> tensor<2x2xi32>
  check.expect_eq_const(%res, dense<24> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

func.func @i16i16.i32() {
  %lhs = util.unfoldable_constant dense<3> : tensor<2x4xi16>
  %rhs = util.unfoldable_constant dense<2> : tensor<4x2xi16>
  %res = "stablehlo.dot"(%lhs, %rhs) : (tensor<2x4xi16>, tensor<4x2xi16>) -> tensor<2x2xi32>
  check.expect_eq_const(%res, dense<24> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

func.func @large() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<15x16xf32>
  %rhs = util.unfoldable_constant dense<0.4> : tensor<16x17xf32>
  %res = "stablehlo.dot"(%lhs, %rhs) : (tensor<15x16xf32>, tensor<16x17xf32>) -> tensor<15x17xf32>
  check.expect_almost_eq_const(%res, dense<6.4> : tensor<15x17xf32>) : tensor<15x17xf32>
  return
}

func.func @matvec() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<15x32xf32>
  %rhs = util.unfoldable_constant dense<0.5> : tensor<32xf32>
  %res = "stablehlo.dot"(%lhs, %rhs) : (tensor<15x32xf32>, tensor<32xf32>) -> tensor<15xf32>
  check.expect_almost_eq_const(%res, dense<16.0> : tensor<15xf32>) : tensor<15xf32>
  return
}

func.func @dot() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<1024xf32>
  %rhs = util.unfoldable_constant dense<0.5> : tensor<1024xf32>
  %res = "stablehlo.dot"(%lhs, %rhs) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<f32>
  check.expect_almost_eq_const(%res, dense<512.0> : tensor<f32>) : tensor<f32>
  return
}
