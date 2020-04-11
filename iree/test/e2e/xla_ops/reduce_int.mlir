// Int sum values from [1, 10]
func @reduce_sum_1x10xi32() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<1x10xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xi32>, tensor<i32>) -> tensor<1xi32>
  check.expect_eq_const(%res, dense<55> : tensor<1xi32>) : tensor<1xi32>
  return
}

// Int max values from [1, 10]
func @reduce_max_1x10xi32() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<1x10xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.maximum"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xi32>, tensor<i32>) -> tensor<1xi32>
  check.expect_eq_const(%res, dense<10> : tensor<1xi32>) : tensor<1xi32>
  return
}

// Int min values, along multiple dimensions. Expected to just be a reshape in this case.
func @reduce_min_5x1x1xi32() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[[1]],[[2]],[[3]],[[4]],[[5]]]> : tensor<5x1x1xi32>
  %1 = iree.unfoldable_constant dense<999> : tensor<i32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.minimum"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<5x1x1xi32>, tensor<i32>) -> tensor<5xi32>
  check.expect_eq_const(%res, dense<[1, 2, 3, 4, 5]> : tensor<5xi32>) : tensor<5xi32>
  return
}


// The following cases match the examples presented at
// https://www.tensorflow.org/xla/operation_semantics#reduce

func @reduce_sum_2x3xi32_dim0() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[
      [1, 2, 3],
      [4, 5, 6]]> : tensor<2x3xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2x3xi32>, tensor<i32>) -> tensor<3xi32>
  check.expect_eq_const(%res, dense<[5, 7, 9]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func @reduce_sum_2x3xi32_dim1() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[
      [1, 2, 3],
      [4, 5, 6]]> : tensor<2x3xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>
  check.expect_eq_const(%res, dense<[6, 15]> : tensor<2xi32>) : tensor<2xi32>
  return
}

func @reduce_sum_4x2x3xi32_dim0() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]]]> : tensor<4x2x3xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<4x2x3xi32>, tensor<i32>) -> tensor<2x3xi32>
  check.expect_eq_const(%res, dense<[[4, 8, 12],[16, 20, 24]]> : tensor<2x3xi32>) : tensor<2x3xi32>
  return
}

func @reduce_sum_4x2x3xi32_dim2() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[
    [[1, 2, 3], [4, 5, 6]],
    [[1, 2, 3], [4, 5, 6]],
    [[1, 2, 3], [4, 5, 6]],
    [[1, 2, 3], [4, 5, 6]]]> : tensor<4x2x3xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<4x2x3xi32>, tensor<i32>) -> tensor<4x2xi32>
  check.expect_eq_const(%res, dense<[[6, 15],[6, 15],[6, 15],[6, 15]]> : tensor<4x2xi32>) : tensor<4x2xi32>
  return
}

func @reduce_sum_4x2x3xi32_dims_0_1() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]]]> : tensor<4x2x3xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x2x3xi32>, tensor<i32>) -> tensor<3xi32>
  check.expect_eq_const(%res, dense<[20, 28, 36]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func @reduce_sum_4x2x3xi32_dims_0_1_2() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]],
      [[1, 2, 3], [4, 5, 6]]]> : tensor<4x2x3xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "xla_hlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<4x2x3xi32>, tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%res, dense<84> : tensor<i32>) : tensor<i32>
  return
}
