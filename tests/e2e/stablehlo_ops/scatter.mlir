func.func @scatter_update_scalar_1D() {
  %arg0 = util.unfoldable_constant dense<0> : tensor<8xi32>
  %arg1 = util.unfoldable_constant dense<[[1], [3], [4], [7]]> : tensor<4x1xi32>
  %arg2 = util.unfoldable_constant dense<[9, 10, 11, 12]> : tensor<4xi32>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    "stablehlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = true
  } : (tensor<8xi32>, tensor<4x1xi32>, tensor<4xi32>) -> tensor<8xi32>
  check.expect_eq_const(%0, dense<[0, 9, 0, 10, 11, 0, 0, 12]> : tensor<8xi32>) : tensor<8xi32>
  return
}

func.func @scatter_repeated_update_scalar_1D() {
  %arg0 = util.unfoldable_constant dense<0> : tensor<8xi32>
  %arg1 = util.unfoldable_constant dense<[[1], [1], [7], [7]]> : tensor<4x1xi32>
  %arg2 = util.unfoldable_constant dense<[9, 10, 11, 12]> : tensor<4xi32>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    "stablehlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false
  } : (tensor<8xi32>, tensor<4x1xi32>, tensor<4xi32>) -> tensor<8xi32>
  check.expect_eq_const(%0, dense<[0, 10, 0, 0, 0, 0, 0, 12]> : tensor<8xi32>) : tensor<8xi32>
  return
}

func.func @scatter_update_scalar_2D() {
  %arg0 = util.unfoldable_constant dense<0> : tensor<4x3xi32>
  %arg1 = util.unfoldable_constant dense<[[0, 0], [1, 1], [2, 2]]> : tensor<3x2xi32>
  %arg2 = util.unfoldable_constant dense<[1, 2, 3]> : tensor<3xi32>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    "stablehlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {indices_are_sorted = false,
      scatter_dimension_numbers = #stablehlo.scatter<
        inserted_window_dims = [0, 1],
        scatter_dims_to_operand_dims = [0, 1],
        index_vector_dim = 1
      >,
      unique_indices = true
  } : (tensor<4x3xi32>, tensor<3x2xi32>, tensor<3xi32>) -> tensor<4x3xi32>
  check.expect_eq_const(%0, dense<[[1, 0, 0],
                                   [0, 2, 0],
                                   [0, 0, 3],
                                   [0, 0, 0]]> : tensor<4x3xi32>) : tensor<4x3xi32>
  return
}

func.func @scatter_update_slice_2D() {
  %arg0 = util.unfoldable_constant dense<0> : tensor<6x3xi32>
  %arg1 = util.unfoldable_constant dense<[[2], [4]]> : tensor<2x1xi32>
  %arg2 = util.unfoldable_constant dense<[[1, 2, 3],
                                          [4, 5, 6]]> : tensor<2x3xi32>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    "stablehlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = true
  } : (tensor<6x3xi32>, tensor<2x1xi32>, tensor<2x3xi32>) -> tensor<6x3xi32>
  check.expect_eq_const(%0, dense<[[0, 0, 0],
                                   [0, 0, 0],
                                   [1, 2, 3],
                                   [0, 0, 0],
                                   [4, 5, 6],
                                   [0, 0, 0]]> : tensor<6x3xi32>) : tensor<6x3xi32>
  return
}

func.func @scatter_update_slice_partial_2D() {
  %arg0 = util.unfoldable_constant dense<0> : tensor<6x3xi32>
  %arg1 = util.unfoldable_constant dense<[[2], [4]]> : tensor<2x1xi32>
  %arg2 = util.unfoldable_constant dense<[[1, 2],
                                          [4, 5]]> : tensor<2x2xi32>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    "stablehlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = true
  } : (tensor<6x3xi32>, tensor<2x1xi32>, tensor<2x2xi32>) -> tensor<6x3xi32>
  check.expect_eq_const(%0, dense<[[0, 0, 0],
                                   [0, 0, 0],
                                   [1, 2, 0],
                                   [0, 0, 0],
                                   [4, 5, 0],
                                   [0, 0, 0]]> : tensor<6x3xi32>) : tensor<6x3xi32>
  return
}

func.func @scatter_add_slice_2D() {
  %arg0 = util.unfoldable_constant dense<1> : tensor<6x3xi32>
  %arg1 = util.unfoldable_constant dense<[[2], [4]]> : tensor<2x1xi32>
  %arg2 = util.unfoldable_constant dense<[[1, 2, 3],
                                          [4, 5, 6]]> : tensor<2x3xi32>
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
    unique_indices = true
  } : (tensor<6x3xi32>, tensor<2x1xi32>, tensor<2x3xi32>) -> tensor<6x3xi32>
  check.expect_eq_const(%0, dense<[[1, 1, 1],
                                   [1, 1, 1],
                                   [2, 3, 4],
                                   [1, 1, 1],
                                   [5, 6, 7],
                                   [1, 1, 1]]> : tensor<6x3xi32>) : tensor<6x3xi32>
  return
}

func.func @scatter_1D_large() {
  %original = util.unfoldable_constant dense<1> : tensor<1400xi32>
  %update = util.unfoldable_constant dense<2> : tensor<1400xi32>
  %init = tensor.empty() : tensor<1400xi32>
  %indices = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%init : tensor<1400xi32>) {
      ^bb0(%arg0: i32):
        %0 = linalg.index 0 : index
     %1 = arith.index_cast %0 : index to i32
     linalg.yield %1 : i32
      } -> tensor<1400xi32>
  %indices_reshaped = tensor.expand_shape %indices [[0, 1]] :
      tensor<1400xi32> into tensor<1400x1xi32>
  %result = "stablehlo.scatter"(%original, %indices_reshaped, %update)({
    ^bb0(%arg3 : tensor<i32>, %arg4 : tensor<i32>):
      "stablehlo.return"(%arg4) : (tensor<i32>) -> ()
    }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = true
  } : (tensor<1400xi32>, tensor<1400x1xi32>, tensor<1400xi32>) -> tensor<1400xi32>
  check.expect_eq_const(%result, dense<2> : tensor<1400xi32>) : tensor<1400xi32>
  return
}

func.func @scatter_2D_large() {
  %original = util.unfoldable_constant dense<1> : tensor<200x300xi32>
  %update = util.unfoldable_constant dense<2> : tensor<200x300xi32>
  %init = tensor.empty() : tensor<200xi32>
  %indices = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%init : tensor<200xi32>) {
      ^bb0(%arg0: i32):
        %0 = linalg.index 0 : index
        %1 = arith.index_cast %0 : index to i32
        linalg.yield %1 : i32
      } -> tensor<200xi32>
  %indices_reshaped = tensor.expand_shape %indices [[0, 1]] :
      tensor<200xi32> into tensor<200x1xi32>
  %result = "stablehlo.scatter"(%original, %indices_reshaped, %update)({
    ^bb0(%arg3 : tensor<i32>, %arg4 : tensor<i32>):
      "stablehlo.return"(%arg4) : (tensor<i32>) -> ()
    }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = true
  } : (tensor<200x300xi32>, tensor<200x1xi32>, tensor<200x300xi32>) -> tensor<200x300xi32>
  check.expect_eq_const(%result, dense<2> : tensor<200x300xi32>) : tensor<200x300xi32>
  return
}

func.func @scatter_2D_large_permuted() {
  %original = util.unfoldable_constant dense<1> : tensor<200x300xi32>
  %update = util.unfoldable_constant dense<2> : tensor<300x200xi32>
  %init = tensor.empty() : tensor<300xi32>
  %indices = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%init : tensor<300xi32>) {
      ^bb0(%arg0: i32):
        %0 = linalg.index 0 : index
        %1 = arith.index_cast %0 : index to i32
        linalg.yield %1 : i32
      } -> tensor<300xi32>
  %indices_reshaped = tensor.expand_shape %indices [[0, 1]] :
      tensor<300xi32> into tensor<300x1xi32>
  %result = "stablehlo.scatter"(%original, %indices_reshaped, %update)({
    ^bb0(%arg3 : tensor<i32>, %arg4 : tensor<i32>):
      "stablehlo.return"(%arg4) : (tensor<i32>) -> ()
    }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [1],
      scatter_dims_to_operand_dims = [1],
      index_vector_dim = 1,
    >,
    unique_indices = true
  } : (tensor<200x300xi32>, tensor<300x1xi32>, tensor<300x200xi32>) -> tensor<200x300xi32>
  check.expect_eq_const(%result, dense<2> : tensor<200x300xi32>) : tensor<200x300xi32>
  return
}
