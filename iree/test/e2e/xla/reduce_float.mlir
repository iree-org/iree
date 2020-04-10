// Float sum values from [1.0, 10.0]
func @reduce_sum_1x10xf32() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]> : tensor<1x10xf32>
  %1 = iree.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
  check.expect_almost_eq_const(%res, dense<55.0> : tensor<1xf32>) : tensor<1xf32>
  return
}

// Float max values from [1.0, 10.0]
func @reduce_max_1x10xf32() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]> : tensor<1x10xf32>
  %1 = iree.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "xla_hlo.reduce"(%0, %1)
  ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
      %3 = "xla_hlo.maximum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  })
  {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
  check.expect_almost_eq_const(%res, dense<10.0> : tensor<1xf32>) : tensor<1xf32>
  return
}

// Float min values, along multiple dimensions. Expected to just be a reshape in this case.
func @reduce_min_5x1x1xf32() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[[1.0]],[[2.0]],[[3.0]],[[4.0]],[[5.0]]]> : tensor<5x1x1xf32>
  %1 = iree.unfoldable_constant dense<999.0> : tensor<f32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
      %3 = "xla_hlo.minimum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<5x1x1xf32>, tensor<f32>) -> tensor<5xf32>
  check.expect_almost_eq_const(%res, dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf32>) : tensor<5xf32>
  return
}

// The following cases match the examples presented at
// https://www.tensorflow.org/xla/operation_semantics#reduce

func @reduce_sum_2x3xf32_dim0() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %1 = iree.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2x3xf32>, tensor<f32>) -> tensor<3xf32>
  check.expect_almost_eq_const(%res, dense<[5.0, 7.0, 9.0]> : tensor<3xf32>) : tensor<3xf32>
  return
}

func @reduce_sum_2x3xf32_dim1() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %1 = iree.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
  check.expect_almost_eq_const(%res, dense<[6.0, 15.0]> : tensor<2xf32>) : tensor<2xf32>
  return
}

func @reduce_sum_4x2x3xf32_dim0() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]> : tensor<4x2x3xf32>
  %1 = iree.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<4x2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  check.expect_almost_eq_const(%res, dense<[[4.0, 8.0, 12.0],[16.0, 20.0, 24.0]]> : tensor<2x3xf32>) : tensor<2x3xf32>
  return
}

func @reduce_sum_4x2x3xf32_dim1() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]> : tensor<4x2x3xf32>
  %1 = iree.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x2x3xf32>, tensor<f32>) -> tensor<4x3xf32>
  check.expect_almost_eq_const(%res, dense<[
      [5.0, 7.0, 9.0],
      [5.0, 7.0, 9.0],
      [5.0, 7.0, 9.0],
      [5.0, 7.0, 9.0]]> : tensor<4x3xf32>) : tensor<4x3xf32>
  return
}

func @reduce_sum_4x2x3xf32_dim2() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]> : tensor<4x2x3xf32>
  %1 = iree.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<4x2x3xf32>, tensor<f32>) -> tensor<4x2xf32>
  check.expect_almost_eq_const(%res, dense<[
      [6.0, 15.0],
      [6.0, 15.0],
      [6.0, 15.0],
      [6.0, 15.0]]> : tensor<4x2xf32>) : tensor<4x2xf32>
  return
}

func @reduce_sum_4x2x3xf32_dims_0_1() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]> : tensor<4x2x3xf32>
  %1 = iree.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x2x3xf32>, tensor<f32>) -> tensor<3xf32>
  check.expect_almost_eq_const(%res, dense<[20.0, 28.0, 36.0]> : tensor<3xf32>) : tensor<3xf32>
  return
}

func @reduce_sum_4x2x3xf32_dims_0_1_2() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]> : tensor<4x2x3xf32>
  %1 = iree.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "xla_hlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<4x2x3xf32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%res, dense<84.0> : tensor<f32>) : tensor<f32>
  return
}
