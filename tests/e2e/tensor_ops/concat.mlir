func.func @concat_i8_static_dim0() {
  %cst_1 = arith.constant dense<1> : tensor<1xi8>
  %cst_2 = arith.constant dense<2> : tensor<1xi8>
  %1 = util.optimization_barrier %cst_1 : tensor<1xi8>
  %2 = util.optimization_barrier %cst_2 : tensor<1xi8>
  %concat = tensor.concat dim(0) %1, %2 : (tensor<1xi8>, tensor<1xi8>) -> tensor<2xi8>
  check.expect_eq_const(%concat, dense<[1,2]> : tensor<2xi8>) : tensor<2xi8>
  return
}

func.func @concat_i16_static_dim0() {
  %cst_1 = arith.constant dense<1> : tensor<1xi16>
  %cst_2 = arith.constant dense<2> : tensor<1xi16>
  %1 = util.optimization_barrier %cst_1 : tensor<1xi16>
  %2 = util.optimization_barrier %cst_2 : tensor<1xi16>
  %concat = tensor.concat dim(0) %1, %2 : (tensor<1xi16>, tensor<1xi16>) -> tensor<2xi16>
  check.expect_eq_const(%concat, dense<[1,2]> : tensor<2xi16>) : tensor<2xi16>
  return
}

func.func @concat_i32_static_dim0() {
  %cst_1 = arith.constant dense<1> : tensor<1xi32>
  %cst_2 = arith.constant dense<2> : tensor<1xi32>
  %1 = util.optimization_barrier %cst_1 : tensor<1xi32>
  %2 = util.optimization_barrier %cst_2 : tensor<1xi32>
  %concat = tensor.concat dim(0) %1, %2 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  check.expect_eq_const(%concat, dense<[1,2]> : tensor<2xi32>) : tensor<2xi32>
  return
}

func.func @concat_i64_static_dim0() {
  %cst_1 = arith.constant dense<1> : tensor<1xi64>
  %cst_2 = arith.constant dense<2> : tensor<1xi64>
  %1 = util.optimization_barrier %cst_1 : tensor<1xi64>
  %2 = util.optimization_barrier %cst_2 : tensor<1xi64>
  %concat = tensor.concat dim(0) %1, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  check.expect_eq_const(%concat, dense<[1,2]> : tensor<2xi64>) : tensor<2xi64>
  return
}

func.func @concat_f32_static_dim0() {
  %cst_1 = arith.constant dense<1.0> : tensor<1xf32>
  %cst_2 = arith.constant dense<2.0> : tensor<1xf32>
  %1 = util.optimization_barrier %cst_1 : tensor<1xf32>
  %2 = util.optimization_barrier %cst_2 : tensor<1xf32>
  %concat = tensor.concat dim(0) %1, %2 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
  check.expect_almost_eq_const(%concat, dense<[1.0,2.0]> : tensor<2xf32>) : tensor<2xf32>
  return
}

func.func @concat_i32_dim1() {
  %lhs = arith.constant dense<[[1,2,3],[-1,-2,-3]]> : tensor<2x3xi32>
  %rhs = arith.constant dense<[[4,5,6,7,8],[-4,-5,-6,-7,-8]]> : tensor<2x5xi32>
  %lhs_barrier = util.optimization_barrier %lhs : tensor<2x3xi32>
  %rhs_barrier = util.optimization_barrier %rhs : tensor<2x5xi32>
  %concat = tensor.concat dim(1) %lhs_barrier, %rhs_barrier : (tensor<2x3xi32>, tensor<2x5xi32>) -> tensor<2x8xi32>
  check.expect_eq_const(%concat, dense<[[1,2,3,4,5,6,7,8],[-1,-2,-3,-4,-5,-6,-7,-8]]> : tensor<2x8xi32>) : tensor<2x8xi32>
  return
}
