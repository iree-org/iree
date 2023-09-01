func.func @scatter_add_slice_2D_dynamic_num_updates() {
  %arg0 = util.unfoldable_constant dense<1> : tensor<6x3xi32>
  %arg1 = flow.tensor.constant dense<[[2], [4]]> : tensor<2x1xi32> -> tensor<?x1xi32>
  %arg2 = flow.tensor.constant dense<[[1, 2, 3],
                                             [4, 5, 6]]> : tensor<2x3xi32> -> tensor<?x3xi32>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    %1 = stablehlo.add %arg3, %arg4 : tensor<i32>
    "stablehlo.return"(%1) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false
  } : (tensor<6x3xi32>, tensor<?x1xi32>, tensor<?x3xi32>) -> tensor<6x3xi32>
  check.expect_eq_const(%0, dense<[[1, 1, 1],
                                   [1, 1, 1],
                                   [2, 3, 4],
                                   [1, 1, 1],
                                   [5, 6, 7],
                                   [1, 1, 1]]> : tensor<6x3xi32>) : tensor<6x3xi32>
  return
}

