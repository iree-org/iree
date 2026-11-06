func.func @dynamic_update_slice_2x2() {
  %target = util.unfoldable_constant dense<2> : tensor<3x3xi32>
  %update = util.unfoldable_constant dense<1> : tensor<2x2xi32>
  %c0 = util.unfoldable_constant dense<0> : tensor<i32>
  %result = "stablehlo.dynamic_update_slice"(%target, %update, %c0, %c0)
    : (tensor<3x3xi32>, tensor<2x2xi32>, tensor<i32>, tensor<i32>) -> tensor<3x3xi32>
  check.expect_eq_const(%result, dense<[
    [1, 1, 2],
    [1, 1, 2],
    [2, 2, 2]]> : tensor<3x3xi32>) : tensor<3x3xi32>
  return
}

func.func @dynamic_update_slice_1x3() {
  %target = util.unfoldable_constant dense<2> : tensor<3x3xi32>
  %update = util.unfoldable_constant dense<1> : tensor<1x3xi32>
  %c0 = util.unfoldable_constant dense<0> : tensor<i32>
  %c1 = util.unfoldable_constant dense<1> : tensor<i32>
  %result = "stablehlo.dynamic_update_slice"(%target, %update, %c1, %c0)
    : (tensor<3x3xi32>, tensor<1x3xi32>, tensor<i32>, tensor<i32>) -> tensor<3x3xi32>
  check.expect_eq_const(%result, dense<[
    [2, 2, 2],
    [1, 1, 1],
    [2, 2, 2]]> : tensor<3x3xi32>) : tensor<3x3xi32>
  return
}

func.func @into_constant() {
  %update = util.unfoldable_constant dense<2> : tensor<1xi32>
  %target = stablehlo.constant dense<1> : tensor<4xi32>
  %index = stablehlo.constant dense<0> : tensor<i32>
  %result = "stablehlo.dynamic_update_slice"(%target, %update, %index) : (tensor<4xi32>, tensor<1xi32>, tensor<i32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[2, 1, 1, 1]> : tensor<4xi32>) : tensor<4xi32>
  return
}

// A run-time row count; the start (3, 4) clamps to (2, 2).
func.func @dynamic_target() {
  %target = flow.tensor.dynamic_constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0, 19.0]]> : tensor<4x5xf32> -> tensor<?x5xf32>
  %update = util.unfoldable_constant dense<[[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]]> : tensor<2x3xf32>
  %i = util.unfoldable_constant dense<3> : tensor<i64>
  %j = util.unfoldable_constant dense<4> : tensor<i64>
  %result = "stablehlo.dynamic_update_slice"(%target, %update, %i, %j)
    : (tensor<?x5xf32>, tensor<2x3xf32>, tensor<i64>, tensor<i64>) -> tensor<?x5xf32>
  %static = tensor.cast %result : tensor<?x5xf32> to tensor<4x5xf32>
  check.expect_eq_const(%static, dense<[
    [0.0, 1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0, 9.0],
    [10.0, 11.0, -1.0, -2.0, -3.0],
    [15.0, 16.0, -4.0, -5.0, -6.0]]> : tensor<4x5xf32>) : tensor<4x5xf32>
  return
}

// A run-time update width.
func.func @dynamic_update() {
  %target = util.unfoldable_constant dense<0> : tensor<3x4xi32>
  %update = flow.tensor.dynamic_constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32> -> tensor<2x?xi32>
  %i = util.unfoldable_constant dense<1> : tensor<i64>
  %j = util.unfoldable_constant dense<1> : tensor<i64>
  %result = "stablehlo.dynamic_update_slice"(%target, %update, %i, %j)
    : (tensor<3x4xi32>, tensor<2x?xi32>, tensor<i64>, tensor<i64>) -> tensor<3x4xi32>
  check.expect_eq_const(%result, dense<[
    [0, 0, 0, 0],
    [0, 1, 2, 0],
    [0, 3, 4, 0]]> : tensor<3x4xi32>) : tensor<3x4xi32>
  return
}
