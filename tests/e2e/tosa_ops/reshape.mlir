func.func @tensor_downrank() {
  %0 = util.unfoldable_constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %1 = tosa.const_shape {values = dense<[4]> : tensor<1xindex>} : () -> !tosa.shape<1>
  %result = tosa.reshape %0, %1 : (tensor<2x2xi32>, !tosa.shape<1>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[1, 2, 3, 4]> : tensor<4xi32>) : tensor<4xi32>
  return
}

func.func @tensor_uprank() {
  %0 = util.unfoldable_constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %1 = tosa.const_shape {values = dense<[2, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %result = tosa.reshape %0, %1 : (tensor<4xi32>, !tosa.shape<2>) -> tensor<2x2xi32>
  check.expect_eq_const(%result, dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

func.func @tensor_crossrank() {
  %0 = util.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %1 = tosa.const_shape {values = dense<[3, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %result = tosa.reshape %0, %1 : (tensor<2x3xi32>, !tosa.shape<2>) -> tensor<3x2xi32>
  check.expect_eq_const(%result, dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>) : tensor<3x2xi32>
  return
}
