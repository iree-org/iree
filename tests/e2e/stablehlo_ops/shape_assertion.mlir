func.func @tensor() {
  %0 = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %1 = util.unfoldable_constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>
  %size = stablehlo.get_dimension_size %0, dim = 0 : (tensor<4xf32>) -> tensor<i32>
  %expected_size = util.unfoldable_constant dense<4> : tensor<i32>
  %predicate = stablehlo.compare EQ, %size, %expected_size, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%predicate) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<i1>) -> ()
  %result = "stablehlo.add"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[6.0, 8.0, 10.0, 12.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func.func @asserted_broadcast_dims() {
  %lhs = flow.tensor.dynamic_constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                                               [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]> : tensor<2x8xf32> -> tensor<?x8xf32>
  %rhs = flow.tensor.dynamic_constant dense<[100.0, 200.0]> : tensor<2xf32> -> tensor<?xf32>
  %lhs_size = stablehlo.get_dimension_size %lhs, dim = 0 : (tensor<?x8xf32>) -> tensor<i32>
  %rhs_size = stablehlo.get_dimension_size %rhs, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %equal = stablehlo.compare EQ, %lhs_size, %rhs_size : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%equal) {error_message = "batch dimensions must match", has_side_effect = true} : (tensor<i1>) -> ()
  %eight = stablehlo.constant dense<8> : tensor<1xi32>
  %batch = stablehlo.reshape %lhs_size : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %batch, %eight, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %broadcast = stablehlo.dynamic_broadcast_in_dim %rhs, %shape, dims = [0] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x8xf32>
  %result = stablehlo.add %lhs, %broadcast : tensor<?x8xf32>
  %expected = flow.tensor.dynamic_constant dense<[[100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
                                                    [208.0, 209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0]]> : tensor<2x8xf32> -> tensor<?x8xf32>
  check.expect_eq(%result, %expected) : tensor<?x8xf32>
  return
}

// Keep the helper boundary so shape scalarisation sees the unsigned operands.
func.func private @unsigned_shape_scalar_helper(%value: ui32) -> (tensor<i1>, tensor<ui32>) attributes {no_inline} {
  %lhs = tensor.from_elements %value : tensor<ui32>
  %zero = stablehlo.constant dense<0> : tensor<ui32>
  %two = stablehlo.constant dense<2> : tensor<ui32>
  %positive = stablehlo.compare GT, %lhs, %zero : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
  %remainder = stablehlo.remainder %lhs, %two : tensor<ui32>
  return %positive, %remainder : tensor<i1>, tensor<ui32>
}

func.func @unsigned_shape_scalar_arithmetic() {
  %bits = util.unfoldable_constant dense<-1> : tensor<i32>
  %input = stablehlo.convert %bits : (tensor<i32>) -> tensor<ui32>
  %value = tensor.extract %input[] : tensor<ui32>
  %positive, %remainder = call @unsigned_shape_scalar_helper(%value) <{no_inline}> : (ui32) -> (tensor<i1>, tensor<ui32>)
  stablehlo.custom_call @shape_assertion(%positive) {error_message = "unsigned high-bit value must be positive", has_side_effect = true} : (tensor<i1>) -> ()
  %result = stablehlo.convert %remainder : (tensor<ui32>) -> tensor<i32>
  check.expect_eq_const(%result, dense<1> : tensor<i32>) : tensor<i32>
  return
}
