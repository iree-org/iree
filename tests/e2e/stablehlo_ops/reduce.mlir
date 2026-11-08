// Int sum values from [1, 10]
func.func @reduce_sum_1x10xi32() {
  %0 = util.unfoldable_constant dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<1x10xi32>
  %1 = util.unfoldable_constant dense<0> : tensor<i32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<1x10xi32>, tensor<i32>) -> tensor<1xi32>
  check.expect_eq_const(%res, dense<55> : tensor<1xi32>) : tensor<1xi32>
  return
}

// Int max values from [1, 10]
func.func @reduce_max_1x10xi32() {
  %0 = util.unfoldable_constant dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<1x10xi32>
  %1 = util.unfoldable_constant dense<0> : tensor<i32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %3 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<1x10xi32>, tensor<i32>) -> tensor<1xi32>
  check.expect_eq_const(%res, dense<10> : tensor<1xi32>) : tensor<1xi32>
  return
}

// Int min values, along multiple dimensions. Expected to just be a reshape in this case.
func.func @reduce_min_5x1x1xi32() {
  %0 = util.unfoldable_constant dense<[[[1]],[[2]],[[3]],[[4]],[[5]]]> : tensor<5x1x1xi32>
  %1 = util.unfoldable_constant dense<999> : tensor<i32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %3 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 1, 2>} : (tensor<5x1x1xi32>, tensor<i32>) -> tensor<5xi32>
  check.expect_eq_const(%res, dense<[1, 2, 3, 4, 5]> : tensor<5xi32>) : tensor<5xi32>
  return
}


// The following cases match the examples presented at
// https://www.tensorflow.org/xla/operation_semantics#reduce

func.func @reduce_sum_2x3xi32_dim0() {
  %0 = util.unfoldable_constant dense<[
      [1, 2, 3],
      [4, 5, 6]]> : tensor<2x3xi32>
  %1 = util.unfoldable_constant dense<0> : tensor<i32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<2x3xi32>, tensor<i32>) -> tensor<3xi32>
  check.expect_eq_const(%res, dense<[5, 7, 9]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func.func @reduce_sum_2x3xi32_dim1() {
  %0 = util.unfoldable_constant dense<[
      [1, 2, 3],
      [4, 5, 6]]> : tensor<2x3xi32>
  %1 = util.unfoldable_constant dense<0> : tensor<i32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>
  check.expect_eq_const(%res, dense<[6, 15]> : tensor<2xi32>) : tensor<2xi32>
  return
}

func.func @reduce_sum_4x2x3xi32_dim0() {
  %0 = util.unfoldable_constant dense<[
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]]]> : tensor<4x2x3xi32>
  %1 = util.unfoldable_constant dense<0> : tensor<i32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<4x2x3xi32>, tensor<i32>) -> tensor<2x3xi32>
  check.expect_eq_const(%res, dense<[[4, 8, 12],[16, 20, 24]]> : tensor<2x3xi32>) : tensor<2x3xi32>
  return
}

func.func @reduce_sum_4x2x3xi32_dim2() {
  %0 = util.unfoldable_constant dense<[
    [[1, 2, 3], [4, 5, 6]],
    [[1, 2, 3], [4, 5, 6]],
    [[1, 2, 3], [4, 5, 6]],
    [[1, 2, 3], [4, 5, 6]]]> : tensor<4x2x3xi32>
  %1 = util.unfoldable_constant dense<0> : tensor<i32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 2>} : (tensor<4x2x3xi32>, tensor<i32>) -> tensor<4x2xi32>
  check.expect_eq_const(%res, dense<[[6, 15],[6, 15],[6, 15],[6, 15]]> : tensor<4x2xi32>) : tensor<4x2xi32>
  return
}

func.func @reduce_sum_4x2x3xi32_dims_0_1() {
  %0 = util.unfoldable_constant dense<[
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]]]> : tensor<4x2x3xi32>
  %1 = util.unfoldable_constant dense<0> : tensor<i32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 0, 1>} : (tensor<4x2x3xi32>, tensor<i32>) -> tensor<3xi32>
  check.expect_eq_const(%res, dense<[20, 28, 36]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func.func @reduce_sum_4x2x3xi32_dims_0_1_2() {
  %0 = util.unfoldable_constant dense<[
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]]]> : tensor<4x2x3xi32>
  %1 = util.unfoldable_constant dense<0> : tensor<i32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 0, 1, 2>} : (tensor<4x2x3xi32>, tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%res, dense<84> : tensor<i32>) : tensor<i32>
  return
}

// Float sum values from [1.0, 10.0]
func.func @reduce_sum_1x10xf32() {
  %0 = util.unfoldable_constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]> : tensor<1x10xf32>
  %1 = util.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
  check.expect_almost_eq_const(%res, dense<55.0> : tensor<1xf32>) : tensor<1xf32>
  return
}

// Float max values from [1.0, 10.0]
func.func @reduce_max_1x10xf32() {
  %0 = util.unfoldable_constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]> : tensor<1x10xf32>
  %1 = util.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "stablehlo.reduce"(%0, %1)
  ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %3 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%3) : (tensor<f32>) -> ()
  })
  {dimensions = array<i64: 1>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
  check.expect_almost_eq_const(%res, dense<10.0> : tensor<1xf32>) : tensor<1xf32>
  return
}

// Float min values, along multiple dimensions. Expected to just be a reshape in this case.
func.func @reduce_min_5x1x1xf32() {
  %0 = util.unfoldable_constant dense<[[[1.0]],[[2.0]],[[3.0]],[[4.0]],[[5.0]]]> : tensor<5x1x1xf32>
  %1 = util.unfoldable_constant dense<999.0> : tensor<f32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %3 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 1, 2>} : (tensor<5x1x1xf32>, tensor<f32>) -> tensor<5xf32>
  check.expect_almost_eq_const(%res, dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf32>) : tensor<5xf32>
  return
}

// The following cases match the examples presented at
// https://www.tensorflow.org/xla/operation_semantics#reduce

func.func @reduce_sum_2x3xf32_dim0() {
  %0 = util.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %1 = util.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<2x3xf32>, tensor<f32>) -> tensor<3xf32>
  check.expect_almost_eq_const(%res, dense<[5.0, 7.0, 9.0]> : tensor<3xf32>) : tensor<3xf32>
  return
}

func.func @reduce_sum_2x3xf32_dim1() {
  %0 = util.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %1 = util.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
  check.expect_almost_eq_const(%res, dense<[6.0, 15.0]> : tensor<2xf32>) : tensor<2xf32>
  return
}

func.func @reduce_sum_4x2x3xf32_dim0() {
  %0 = util.unfoldable_constant dense<[
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]> : tensor<4x2x3xf32>
  %1 = util.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<4x2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  check.expect_almost_eq_const(%res, dense<[[4.0, 8.0, 12.0],[16.0, 20.0, 24.0]]> : tensor<2x3xf32>) : tensor<2x3xf32>
  return
}

func.func @reduce_sum_4x2x3xf32_dim1() {
  %0 = util.unfoldable_constant dense<[
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]> : tensor<4x2x3xf32>
  %1 = util.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<4x2x3xf32>, tensor<f32>) -> tensor<4x3xf32>
  check.expect_almost_eq_const(%res, dense<[
      [5.0, 7.0, 9.0],
      [5.0, 7.0, 9.0],
      [5.0, 7.0, 9.0],
      [5.0, 7.0, 9.0]]> : tensor<4x3xf32>) : tensor<4x3xf32>
  return
}

func.func @reduce_sum_4x2x3xf32_dim2() {
  %0 = util.unfoldable_constant dense<[
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]> : tensor<4x2x3xf32>
  %1 = util.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 2>} : (tensor<4x2x3xf32>, tensor<f32>) -> tensor<4x2xf32>
  check.expect_almost_eq_const(%res, dense<[
      [6.0, 15.0],
      [6.0, 15.0],
      [6.0, 15.0],
      [6.0, 15.0]]> : tensor<4x2xf32>) : tensor<4x2xf32>
  return
}

func.func @reduce_sum_4x2x3xf32_dims_0_1() {
  %0 = util.unfoldable_constant dense<[
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]> : tensor<4x2x3xf32>
  %1 = util.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 0, 1>} : (tensor<4x2x3xf32>, tensor<f32>) -> tensor<3xf32>
  check.expect_almost_eq_const(%res, dense<[20.0, 28.0, 36.0]> : tensor<3xf32>) : tensor<3xf32>
  return
}

func.func @reduce_sum_4x2x3xf32_dims_0_1_2() {
  %0 = util.unfoldable_constant dense<[
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]> : tensor<4x2x3xf32>
  %1 = util.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 0, 1, 2>} : (tensor<4x2x3xf32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%res, dense<84.0> : tensor<f32>) : tensor<f32>
  return
}

func.func @reducemulti_result() {
  %cst0 = stablehlo.constant dense<-2147483648> : tensor<i32>
  %cst1 = stablehlo.constant dense<0> : tensor<i32>
  %arg0 = util.unfoldable_constant dense<[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]]> : tensor<9x2xi32>
  %arg1 = util.unfoldable_constant dense<[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17]]> : tensor<9x2xi32>
  %res0, %res1 = "stablehlo.reduce"(%arg0, %arg1, %cst0, %cst1) ( {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
    %0 = "stablehlo.compare"(%arg2, %arg4) {comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1 = "stablehlo.select"(%0, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = "stablehlo.compare"(%arg2, %arg4) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = stablehlo.minimum %arg3, %arg5 : tensor<i32>
    %4 = "stablehlo.select"(%0, %arg3, %arg5) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = "stablehlo.select"(%2, %3, %4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%1, %5) : (tensor<i32>, tensor<i32>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<9x2xi32>, tensor<9x2xi32>, tensor<i32>, tensor<i32>) -> (tensor<2xi32>, tensor<2xi32>)
  check.expect_eq_const(%res0, dense<[17, 18]> : tensor<2xi32>) : tensor<2xi32>
  check.expect_eq_const(%res1, dense<[16, 17]> : tensor<2xi32>) : tensor<2xi32>
  return
}

func.func @reduce_dim_1() {
  %0 = util.unfoldable_constant dense<[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]> : tensor<2x5xi32>
  %1 = util.unfoldable_constant dense<10> : tensor<i32>
  %2 = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0 : tensor<i32>, %arg1 : tensor<i32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<2x5xi32>, tensor<i32>) -> tensor<2xi32>
  check.expect_eq_const(%2, dense<[25, 50]> : tensor<2xi32>) : tensor<2xi32>
  return
}

// Constants get folded in which linalg.indexed_generic ops. Check to
// make sure this works as expected.
func.func @reduce_dim_1_const() {
  %0 = util.unfoldable_constant dense<[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]> : tensor<2x5xi32>
  %1 = arith.constant dense<10> : tensor<i32>
  %2 = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0 : tensor<i32>, %arg1 : tensor<i32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<2x5xi32>, tensor<i32>) -> tensor<2xi32>
  check.expect_eq_const(%2, dense<[25, 50]> : tensor<2xi32>) : tensor<2xi32>
  return
}

func.func @reduce_dim_0() {
  %0 = util.unfoldable_constant dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<1x10xi32>
  %1 = util.unfoldable_constant dense<10> : tensor<i32>
  %2 = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0 : tensor<i32>, %arg1 : tensor<i32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<1x10xi32>, tensor<i32>) -> tensor<1xi32>
  check.expect_eq_const(%2, dense<[65]> : tensor<1xi32>) : tensor<1xi32>
  return
}

func.func @reduce_to_scalar() {
  %0 = util.unfoldable_constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<10xi32>
  %1 = util.unfoldable_constant dense<10> : tensor<i32>
  %2 = "stablehlo.reduce"(%0, %1) ( {
  ^bb0(%arg0 : tensor<i32>, %arg1 : tensor<i32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<10xi32>, tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%2, dense<65> : tensor<i32>) : tensor<i32>
  return
}


// Reduction arithmetic can be wider than its inputs and initializers.
func.func private @promotion_test_promoted_inputs(%p: tensor<2xi1>, %s: tensor<2xi8>, %u: tensor<2xui8>, %init: tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  %a = stablehlo.reduce(%p init: %init) across dimensions = [0] : (tensor<2xi1>, tensor<i32>) -> tensor<i32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %v = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %v : tensor<i32>
    }
  %b = stablehlo.reduce(%s init: %init) across dimensions = [0] : (tensor<2xi8>, tensor<i32>) -> tensor<i32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %v = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %v : tensor<i32>
    }
  %c = stablehlo.reduce(%u init: %init) across dimensions = [0] : (tensor<2xui8>, tensor<i32>) -> tensor<i32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %v = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %v : tensor<i32>
    }
  return %a, %b, %c : tensor<i32>, tensor<i32>, tensor<i32>
}

func.func private @promotion_test_promoted_initializer(%x: tensor<?xf32>, %init: tensor<f16>) -> tensor<f32> {
  %r = stablehlo.reduce(%x init: %init) across dimensions = [0] : (tensor<?xf32>, tensor<f16>) -> tensor<f32>
    reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %v = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %v : tensor<f32>
    }
  return %r : tensor<f32>
}

func.func private @promotion_test_zero_axes_promoted(%x: tensor<2xui8>, %init: tensor<i32>) -> tensor<2xi32> {
  %r = stablehlo.reduce(%x init: %init) across dimensions = [] : (tensor<2xui8>, tensor<i32>) -> tensor<2xi32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %v = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %v : tensor<i32>
    }
  return %r : tensor<2xi32>
}

func.func private @promotion_test_empty_promoted_initializer(%x: tensor<0x3xi8>, %init: tensor<i8>) -> tensor<3xi32> {
  %r = stablehlo.reduce(%x init: %init) across dimensions = [0] : (tensor<0x3xi8>, tensor<i8>) -> tensor<3xi32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %v = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %v : tensor<i32>
    }
  return %r : tensor<3xi32>
}

func.func private @promotion_test_dynamic_empty_promoted_initializer(%x: tensor<?x0xi8>, %init: tensor<i8>) -> tensor<?xi32> {
  %r = stablehlo.reduce(%x init: %init) across dimensions = [1] : (tensor<?x0xi8>, tensor<i8>) -> tensor<?xi32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %v = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %v : tensor<i32>
    }
  return %r : tensor<?xi32>
}

func.func private @promotion_test_variadic_promotions(%s: tensor<2xi8>, %u: tensor<2xui8>, %init0: tensor<i16>, %init1: tensor<i32>) -> (tensor<i32>, tensor<i64>) {
  %r:2 = "stablehlo.reduce"(%s, %u, %init0, %init1) ({
    ^bb0(%a: tensor<i32>, %b: tensor<i64>, %x: tensor<i32>, %y: tensor<i64>):
      %v0 = stablehlo.add %a, %x : tensor<i32>
      %v1 = stablehlo.add %b, %y : tensor<i64>
      stablehlo.return %v0, %v1 : tensor<i32>, tensor<i64>
  }) {dimensions = array<i64: 0>} : (tensor<2xi8>, tensor<2xui8>, tensor<i16>, tensor<i32>) -> (tensor<i32>, tensor<i64>)
  return %r#0, %r#1 : tensor<i32>, tensor<i64>
}

// Predicate widening must use positive one; signed and unsigned bytes differ.
func.func @reduce_promoted_integer_inputs() {
  %p = util.unfoldable_constant dense<true> : tensor<2xi1>
  %s = util.unfoldable_constant dense<[-1, 1]> : tensor<2xi8>
  %u_raw = stablehlo.constant dense<[255, 1]> : tensor<2xui8>
  %u = "stablehlo.optimization_barrier"(%u_raw) : (tensor<2xui8>) -> tensor<2xui8>
  %zero = util.unfoldable_constant dense<0> : tensor<i32>
  %r:3 = func.call @promotion_test_promoted_inputs(%p, %s, %u, %zero) : (tensor<2xi1>, tensor<2xi8>, tensor<2xui8>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
  check.expect_eq_const(%r#0, dense<2> : tensor<i32>) : tensor<i32>
  check.expect_eq_const(%r#1, dense<0> : tensor<i32>) : tensor<i32>
  check.expect_eq_const(%r#2, dense<256> : tensor<i32>) : tensor<i32>
  return
}

func.func @reduce_promoted_float_initializer() {
  %x = util.unfoldable_constant dense<[1.0, 2.0]> : tensor<2xf32>
  %dynamic = tensor.cast %x : tensor<2xf32> to tensor<?xf32>
  %zero = util.unfoldable_constant dense<0.0> : tensor<f16>
  %r = func.call @promotion_test_promoted_initializer(%dynamic, %zero) : (tensor<?xf32>, tensor<f16>) -> tensor<f32>
  check.expect_eq_const(%r, dense<3.0> : tensor<f32>) : tensor<f32>
  return
}

func.func @reduce_promoted_zero_axes() {
  %x_raw = stablehlo.constant dense<[255, 1]> : tensor<2xui8>
  %x = "stablehlo.optimization_barrier"(%x_raw) : (tensor<2xui8>) -> tensor<2xui8>
  %zero = util.unfoldable_constant dense<0> : tensor<i32>
  %r = func.call @promotion_test_zero_axes_promoted(%x, %zero) : (tensor<2xui8>, tensor<i32>) -> tensor<2xi32>
  check.expect_eq_const(%r, dense<[255, 1]> : tensor<2xi32>) : tensor<2xi32>
  return
}

func.func @reduce_promoted_empty_initializer() {
  %x = util.unfoldable_constant dense<> : tensor<0x3xi8>
  %init = util.unfoldable_constant dense<-2> : tensor<i8>
  %r = func.call @promotion_test_empty_promoted_initializer(%x, %init) : (tensor<0x3xi8>, tensor<i8>) -> tensor<3xi32>
  check.expect_eq_const(%r, dense<-2> : tensor<3xi32>) : tensor<3xi32>
  return
}

func.func @reduce_promoted_dynamic_empty_initializer() {
  %x = util.unfoldable_constant dense<> : tensor<2x0xi8>
  %dynamic = tensor.cast %x : tensor<2x0xi8> to tensor<?x0xi8>
  %init = util.unfoldable_constant dense<-2> : tensor<i8>
  %r = func.call @promotion_test_dynamic_empty_promoted_initializer(%dynamic, %init) : (tensor<?x0xi8>, tensor<i8>) -> tensor<?xi32>
  %static = tensor.cast %r : tensor<?xi32> to tensor<2xi32>
  check.expect_eq_const(%static, dense<-2> : tensor<2xi32>) : tensor<2xi32>
  return
}

func.func @reduce_promoted_variadic() {
  %s = util.unfoldable_constant dense<[-1, 1]> : tensor<2xi8>
  %u_raw = stablehlo.constant dense<[255, 1]> : tensor<2xui8>
  %u = "stablehlo.optimization_barrier"(%u_raw) : (tensor<2xui8>) -> tensor<2xui8>
  %init0 = util.unfoldable_constant dense<0> : tensor<i16>
  %init1 = util.unfoldable_constant dense<0> : tensor<i32>
  %r:2 = func.call @promotion_test_variadic_promotions(%s, %u, %init0, %init1) : (tensor<2xi8>, tensor<2xui8>, tensor<i16>, tensor<i32>) -> (tensor<i32>, tensor<i64>)
  check.expect_eq_const(%r#0, dense<0> : tensor<i32>) : tensor<i32>
  check.expect_eq_const(%r#1, dense<256> : tensor<i64>) : tensor<i64>
  return
}
