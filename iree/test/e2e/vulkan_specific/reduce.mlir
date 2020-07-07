func @reduce_dim_1() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]> : tensor<2x5xi32>
  %1 = iree.unfoldable_constant dense<10> : tensor<i32>
  %2 = "mhlo.reduce"(%0, %1) ( {
  ^bb0(%arg0 : tensor<i32>, %arg1 : tensor<i32>):
    %3 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<2x5xi32>, tensor<i32>) -> tensor<2xi32>
  check.expect_eq_const(%2, dense<[25, 50]> : tensor<2xi32>) : tensor<2xi32>
  return
}

// Constants get folded in which linalg.indexed_generic ops. Check to
// make sure this works as expected.
func @reduce_dim_1_const() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]> : tensor<2x5xi32>
  %1 = constant dense<10> : tensor<i32>
  %2 = "mhlo.reduce"(%0, %1) ( {
  ^bb0(%arg0 : tensor<i32>, %arg1 : tensor<i32>):
    %3 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<2x5xi32>, tensor<i32>) -> tensor<2xi32>
  check.expect_eq_const(%2, dense<[25, 50]> : tensor<2xi32>) : tensor<2xi32>
  return
}

func @reduce_dim_0() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<1x10xi32>
  %1 = iree.unfoldable_constant dense<10> : tensor<i32>
  %2 = "mhlo.reduce"(%0, %1) ( {
  ^bb0(%arg0 : tensor<i32>, %arg1 : tensor<i32>):
    %3 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xi32>, tensor<i32>) -> tensor<1xi32>
  check.expect_eq_const(%2, dense<[65]> : tensor<1xi32>) : tensor<1xi32>
  return
}

func @reduce_to_scalar() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<10xi32>
  %1 = iree.unfoldable_constant dense<10> : tensor<i32>
  %2 = "mhlo.reduce"(%0, %1) ( {
  ^bb0(%arg0 : tensor<i32>, %arg1 : tensor<i32>):
    %3 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%3) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<10xi32>, tensor<i32>) -> tensor<i32>
  check.expect_eq_const(%2, dense<65> : tensor<i32>) : tensor<i32>
  return
}
