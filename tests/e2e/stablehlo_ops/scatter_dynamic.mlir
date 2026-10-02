func.func @scatter_add_slice_2D_dynamic_num_updates() {
  %arg0 = util.unfoldable_constant dense<1> : tensor<6x3xi32>
  %arg1 = flow.tensor.dynamic_constant dense<[[2], [4]]> : tensor<2x1xi32> -> tensor<?x1xi32>
  %arg2 = flow.tensor.dynamic_constant dense<[[1, 2, 3],
                                             [4, 5, 6]]> : tensor<2x3xi32> -> tensor<?x3xi32>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
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

// Dynamic batching promotes the indices to i64 before concatenating the iota.
func.func @scatter_dynamic_batch_i32_indices() {
  %a = flow.tensor.dynamic_constant dense<0> : tensor<2x4xi32> -> tensor<?x4xi32>
  %i = flow.tensor.dynamic_constant dense<[[[0], [2], [0]], [[1], [3], [1]]]> : tensor<2x3x1xi32> -> tensor<?x3x1xi32>
  %u = flow.tensor.dynamic_constant dense<[[1, 2, 4], [8, 16, 32]]> : tensor<2x3xi32> -> tensor<?x3xi32>
  %r = "stablehlo.scatter"(%a, %i, %u) ({
  ^bb0(%x: tensor<i32>, %y: tensor<i32>):
    %sum = stablehlo.add %x, %y : tensor<i32>
    "stablehlo.return"(%sum) : (tensor<i32>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [], inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>,
    indices_are_sorted = false, unique_indices = false
  } : (tensor<?x4xi32>, tensor<?x3x1xi32>, tensor<?x3xi32>) -> tensor<?x4xi32>
  %expected = arith.constant dense<[[5, 0, 2, 0], [0, 40, 0, 16]]> : tensor<2x4xi32>
  %expected_dynamic = tensor.cast %expected : tensor<2x4xi32> to tensor<?x4xi32>
  check.expect_eq(%r, %expected_dynamic) : tensor<?x4xi32>
  return
}
