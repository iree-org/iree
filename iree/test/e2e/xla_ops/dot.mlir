func @f32() {
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
  %res = "mhlo.dot"(%lhs, %rhs) : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>
  check.expect_almost_eq_const(%res, dense<[
    [430.0, 388.0, 346.0, 304.0, 262.0],
    [340.0, 307.0, 274.0, 241.0, 208.0],
    [250.0, 226.0, 202.0, 178.0, 154.0],
    [160.0, 145.0, 130.0, 115.0, 100.0],
    [70.0, 64.0, 58.0, 52.0, 46.0]]> : tensor<5x5xf32>) : tensor<5x5xf32>
  return
}

func @i32i32.i32() {
  %lhs = util.unfoldable_constant dense<3> : tensor<2x4xi32>
  %rhs = util.unfoldable_constant dense<2> : tensor<4x2xi32>
  %res = "mhlo.dot"(%lhs, %rhs) : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>
  check.expect_eq_const(%res, dense<24> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

func @i8i8.i32() {
  %lhs = util.unfoldable_constant dense<3> : tensor<2x4xi8>
  %rhs = util.unfoldable_constant dense<2> : tensor<4x2xi8>
  %res = "mhlo.dot"(%lhs, %rhs) : (tensor<2x4xi8>, tensor<4x2xi8>) -> tensor<2x2xi32>
  check.expect_eq_const(%res, dense<24> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

func @i16i16.i32() {
  %lhs = util.unfoldable_constant dense<3> : tensor<2x4xi16>
  %rhs = util.unfoldable_constant dense<2> : tensor<4x2xi16>
  %res = "mhlo.dot"(%lhs, %rhs) : (tensor<2x4xi16>, tensor<4x2xi16>) -> tensor<2x2xi32>
  check.expect_eq_const(%res, dense<24> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

func @large() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<250x1024xf32>
  %rhs = util.unfoldable_constant dense<0.4> : tensor<1024x500xf32>
  %res = "mhlo.dot"(%lhs, %rhs) : (tensor<250x1024xf32>, tensor<1024x500xf32>) -> tensor<250x500xf32>
  check.expect_almost_eq_const(%res, dense<409.596> : tensor<250x500xf32>) : tensor<250x500xf32>
  return
}
