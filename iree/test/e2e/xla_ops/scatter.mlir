func @scatter_scalar() attributes { iree.module.export } {
  %arg0 = iree.unfoldable_constant dense<0> : tensor<8xi32>
  %arg1 = iree.unfoldable_constant dense<[[1], [3], [4], [7]]> : tensor<4x1xi32>
  %arg2 = iree.unfoldable_constant dense<[9, 10, 11, 12]> : tensor<4xi32>
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = {
      index_vector_dim = 1 : i64,
      inserted_window_dims = dense<0> : tensor<1xi64>,
      scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>,
      update_window_dims = dense<> : tensor<0xi64>
    },
    unique_indices = false
  } : (tensor<8xi32>, tensor<4x1xi32>, tensor<4xi32>) -> tensor<8xi32>
  check.expect_eq_const(%0, dense<[0, 9, 0, 10, 11, 0, 0, 12]> : tensor<8xi32>) : tensor<8xi32>
  return
}

func @scatter_slice() attributes { iree.module.export } {
  %arg0 = iree.unfoldable_constant dense<0> : tensor<6x3xi32>
  %arg1 = iree.unfoldable_constant dense<[[2], [4]]> : tensor<2x1xi32>
  %arg2 = iree.unfoldable_constant dense<[[1, 2, 3],
                                          [4, 5, 6]]> : tensor<2x3xi32>
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = {
      index_vector_dim = 1 : i64,
      inserted_window_dims = dense<0> : tensor<1xi64>,
      scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>,
      update_window_dims = dense<1> : tensor<1xi64>
    },
    unique_indices = false
  } : (tensor<6x3xi32>, tensor<2x1xi32>, tensor<2x3xi32>) -> tensor<6x3xi32>
  check.expect_eq_const(%0, dense<[[0, 0, 0],
                                   [0, 0, 0],
                                   [1, 2, 3],
                                   [0, 0, 0],
                                   [4, 5, 6],
                                   [0, 0, 0]]> : tensor<6x3xi32>) : tensor<6x3xi32>
  return
}
