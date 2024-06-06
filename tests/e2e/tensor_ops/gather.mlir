func.func @basic() {
  %source = util.unfoldable_constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>
  %indices = util.unfoldable_constant dense<[[0], [2]]> : tensor<2x1xi32>
  %result = tensor.gather %source[%indices] gather_dims([0]) : (tensor<4x4xi32>, tensor<2x1xi32>) -> tensor<2x4xi32>
  check.expect_eq_const(%result, dense<[[0, 1, 2, 3], [8, 9, 10, 11]]> : tensor<2x4xi32>) : tensor<2x4xi32>
  return
}


func.func @dim_1() {
  %source = util.unfoldable_constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>
  %indices = util.unfoldable_constant dense<[[0], [2]]> : tensor<2x1xi32>
  %result = tensor.gather %source[%indices] gather_dims([1]) : (tensor<4x4xi32>, tensor<2x1xi32>) -> tensor<2x4xi32>
  check.expect_eq_const(%result, dense<[[0, 4, 8, 12], [2, 6, 10, 14]]> : tensor<2x4xi32>) : tensor<2x4xi32>
  return
}

func.func @scalar_3() {
  %source = util.unfoldable_constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>
  %indices = util.unfoldable_constant dense<[[0, 0], [3,3]]> : tensor<2x2xi32>
  %result = tensor.gather %source[%indices] gather_dims([0, 1]) : (tensor<4x4xi32>, tensor<2x2xi32>) -> tensor<2xi32>
  check.expect_eq_const(%result, dense<[0, 15]> : tensor<2xi32>) : tensor<2xi32>
  return
}
