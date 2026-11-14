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

func.func @scatter_dynamic_batch_i32_indices() {
  %operand = flow.tensor.dynamic_constant dense<0> : tensor<2x4xi32> -> tensor<?x4xi32>
  %indices = flow.tensor.dynamic_constant dense<[[[1], [3], [0]], [[2], [1], [3]]]> : tensor<2x3x1xi32> -> tensor<?x3x1xi32>
  %updates = flow.tensor.dynamic_constant dense<[[1, 2, 4], [8, 16, 32]]> : tensor<2x3xi32> -> tensor<?x3xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
    ^bb0(%current: tensor<i32>, %update: tensor<i32>):
      %sum = stablehlo.add %current, %update : tensor<i32>
      stablehlo.return %sum : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [1], input_batching_dims = [0],
      scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1],
      index_vector_dim = 2>
  } : (tensor<?x4xi32>, tensor<?x3x1xi32>, tensor<?x3xi32>) -> tensor<?x4xi32>
  %expected = arith.constant dense<[[4, 1, 0, 2], [0, 16, 8, 32]]> : tensor<2x4xi32>
  %expected_dynamic = tensor.cast %expected : tensor<2x4xi32> to tensor<?x4xi32>
  check.expect_eq(%result, %expected_dynamic) : tensor<?x4xi32>
  return
}

// Boolean true remains index one when dynamic batching generates coordinates.
func.func @scatter_dynamic_batch_bool_indices() {
  %operand = flow.tensor.dynamic_constant dense<0> : tensor<2x4xi32> -> tensor<?x4xi32>
  %indices = flow.tensor.dynamic_constant dense<[[[true], [false], [true]], [[false], [true], [false]]]> : tensor<2x3x1xi1> -> tensor<?x3x1xi1>
  %updates = flow.tensor.dynamic_constant dense<[[1, 2, 4], [8, 16, 32]]> : tensor<2x3xi32> -> tensor<?x3xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
    ^bb0(%current: tensor<i32>, %update: tensor<i32>):
      %sum = stablehlo.add %current, %update : tensor<i32>
      stablehlo.return %sum : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [1], input_batching_dims = [0],
      scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1],
      index_vector_dim = 2>
  } : (tensor<?x4xi32>, tensor<?x3x1xi1>, tensor<?x3xi32>) -> tensor<?x4xi32>
  %expected = arith.constant dense<[[2, 5, 0, 0], [40, 16, 0, 0]]> : tensor<2x4xi32>
  %expected_dynamic = tensor.cast %expected : tensor<2x4xi32> to tensor<?x4xi32>
  check.expect_eq(%result, %expected_dynamic) : tensor<?x4xi32>
  return
}

// The fused generator may read through a widening Boolean conversion.
func.func @scatter_dynamic_batch_widened_bool_indices() {
  %operand = flow.tensor.dynamic_constant dense<0> : tensor<2x4xi32> -> tensor<?x4xi32>
  %indices = flow.tensor.dynamic_constant dense<[[[true], [false], [true]], [[false], [true], [false]]]> : tensor<2x3x1xi1> -> tensor<?x3x1xi1>
  %updates = flow.tensor.dynamic_constant dense<[[1, 2, 4], [8, 16, 32]]> : tensor<2x3xi32> -> tensor<?x3xi32>
  %wide = stablehlo.convert %indices : (tensor<?x3x1xi1>) -> tensor<?x3x1xi32>
  %result = "stablehlo.scatter"(%operand, %wide, %updates) ({
    ^bb0(%current: tensor<i32>, %update: tensor<i32>):
      %sum = stablehlo.add %current, %update : tensor<i32>
      stablehlo.return %sum : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [1], input_batching_dims = [0],
      scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1],
      index_vector_dim = 2>
  } : (tensor<?x4xi32>, tensor<?x3x1xi32>, tensor<?x3xi32>) -> tensor<?x4xi32>
  %expected = arith.constant dense<[[2, 5, 0, 0], [40, 16, 0, 0]]> : tensor<2x4xi32>
  %expected_dynamic = tensor.cast %expected : tensor<2x4xi32> to tensor<?x4xi32>
  check.expect_eq(%result, %expected_dynamic) : tensor<?x4xi32>
  return
}

// The implicit trailing index-vector dimension is expanded dynamically too.
func.func @scatter_dynamic_batch_implicit_index_vector() {
  %operand = flow.tensor.dynamic_constant dense<0> : tensor<2x4xi32> -> tensor<?x4xi32>
  %indices = flow.tensor.dynamic_constant dense<[[1, 3, 0], [2, 1, 3]]> : tensor<2x3xi32> -> tensor<?x3xi32>
  %updates = flow.tensor.dynamic_constant dense<[[1, 2, 4], [8, 16, 32]]> : tensor<2x3xi32> -> tensor<?x3xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
    ^bb0(%current: tensor<i32>, %update: tensor<i32>):
      %sum = stablehlo.add %current, %update : tensor<i32>
      stablehlo.return %sum : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [1], input_batching_dims = [0],
      scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1],
      index_vector_dim = 2>
  } : (tensor<?x4xi32>, tensor<?x3xi32>, tensor<?x3xi32>) -> tensor<?x4xi32>
  %expected = arith.constant dense<[[4, 1, 0, 2], [0, 16, 8, 32]]> : tensor<2x4xi32>
  %expected_dynamic = tensor.cast %expected : tensor<2x4xi32> to tensor<?x4xi32>
  check.expect_eq(%result, %expected_dynamic) : tensor<?x4xi32>
  return
}

// An empty explicit index-vector dimension contains no source coordinates.
// The fused preprocessing must only extract source indices on the nonempty
// source-vector suffix, while still emitting the batch coordinate prefix.
func.func @scatter_dynamic_batch_empty_index_vector() {
  %operand = flow.tensor.dynamic_constant dense<[1, 2]> : tensor<2xi32> -> tensor<?xi32>
  %indices = flow.tensor.dynamic_constant dense<> : tensor<2x0xi32> -> tensor<?x0xi32>
  %updates = flow.tensor.dynamic_constant dense<[9, 8]> : tensor<2xi32> -> tensor<?xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
    ^bb0(%current: tensor<i32>, %update: tensor<i32>):
      stablehlo.return %update : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      input_batching_dims = [0], scatter_indices_batching_dims = [0],
      index_vector_dim = 1>
  } : (tensor<?xi32>, tensor<?x0xi32>, tensor<?xi32>) -> tensor<?xi32>
  %expected = arith.constant dense<[9, 8]> : tensor<2xi32>
  %expected_dynamic = tensor.cast %expected : tensor<2xi32> to tensor<?xi32>
  check.expect_eq(%result, %expected_dynamic) : tensor<?xi32>
  return
}

// An explicit empty index vector has a dynamic batch extent on indices and a
// static extent on updates; the runtime shapes agree.
func.func @scatter_dynamic_indices_static_updates() {
  %operand = arith.constant dense<[1, 2]> : tensor<2xi32>
  %indices = flow.tensor.dynamic_constant dense<> : tensor<2x0xi32> -> tensor<?x0xi32>
  %updates = arith.constant dense<[9, 8]> : tensor<2xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
    ^bb0(%current: tensor<i32>, %update: tensor<i32>):
      stablehlo.return %update : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      input_batching_dims = [0], scatter_indices_batching_dims = [0],
      index_vector_dim = 1>
  } : (tensor<2xi32>, tensor<?x0xi32>, tensor<2xi32>) -> tensor<2xi32>
  %expected = arith.constant dense<[9, 8]> : tensor<2xi32>
  check.expect_eq(%result, %expected) : tensor<2xi32>
  return
}

// The inverse pairing has static indices and dynamic updates with the same
// runtime batch extent.
func.func @scatter_static_indices_dynamic_updates() {
  %operand = arith.constant dense<[1, 2]> : tensor<2xi32>
  %indices = arith.constant dense<> : tensor<2x0xi32>
  %updates = flow.tensor.dynamic_constant dense<[9, 8]> : tensor<2xi32> -> tensor<?xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
    ^bb0(%current: tensor<i32>, %update: tensor<i32>):
      stablehlo.return %update : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      input_batching_dims = [0], scatter_indices_batching_dims = [0],
      index_vector_dim = 1>
  } : (tensor<2xi32>, tensor<2x0xi32>, tensor<?xi32>) -> tensor<2xi32>
  %expected = arith.constant dense<[9, 8]> : tensor<2xi32>
  check.expect_eq(%result, %expected) : tensor<2xi32>
  return
}

// A large static batch with a dynamic update count requires widening i8
// indices, while the generator keeps the converted tensor out of a temporary.
func.func @scatter_dynamic_updates_static_batch() {
  %operand = util.unfoldable_constant dense<0> : tensor<256x4xi32>
  %index_bits = flow.tensor.dynamic_constant dense<1> : tensor<256x1x1xi8> -> tensor<256x?x1xi8>
  %indices = stablehlo.bitcast_convert %index_bits : (tensor<256x?x1xi8>) -> tensor<256x?x1xui8>
  %updates = flow.tensor.dynamic_constant dense<1> : tensor<256x1xi32> -> tensor<256x?xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
    ^bb0(%current: tensor<i32>, %update: tensor<i32>):
      %sum = stablehlo.add %current, %update : tensor<i32>
      stablehlo.return %sum : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [1], input_batching_dims = [0],
      scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1],
      index_vector_dim = 2>
  } : (tensor<256x4xi32>, tensor<256x?x1xui8>, tensor<256x?xi32>) -> tensor<256x4xi32>
  %expected_row = arith.constant dense<[[0, 1, 0, 0]]> : tensor<1x4xi32>
  %expected = stablehlo.broadcast_in_dim %expected_row, dims = [0, 1] : (tensor<1x4xi32>) -> tensor<256x4xi32>
  check.expect_eq(%result, %expected) : tensor<256x4xi32>
  return
}

// Out-of-bounds scatter indices leave the corresponding output unchanged.
func.func @scatter_oob_dynamic_index_batch() {
  %operand = flow.tensor.dynamic_constant dense<[85]> : tensor<1xi32> -> tensor<?xi32>
  %indices = flow.tensor.dynamic_constant dense<[[0], [-2], [-2], [-1], [-2], [-2], [-2], [-1], [1], [0], [1]]> : tensor<11x1xi32> -> tensor<?x1xi32>
  %updates = flow.tensor.dynamic_constant dense<[40, -554, 29, 777, -304, -393, 583, 444, -493, 486, 497]> : tensor<11xi32> -> tensor<?xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
    ^bb0(%current: tensor<i32>, %update: tensor<i32>):
      %sum = stablehlo.add %current, %update : tensor<i32>
      stablehlo.return %sum : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [0], scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>
  } : (tensor<?xi32>, tensor<?x1xi32>, tensor<?xi32>) -> tensor<?xi32>
  %expected = arith.constant dense<[611]> : tensor<1xi32>
  %expected_dynamic = tensor.cast %expected : tensor<1xi32> to tensor<?xi32>
  check.expect_eq(%result, %expected_dynamic) : tensor<?xi32>
  return
}

func.func @scatter_oob_static_index_batch() {
  %operand = util.unfoldable_constant dense<[85]> : tensor<1xi32>
  %indices = util.unfoldable_constant dense<[[0], [-2], [-2], [-1], [-2], [-2], [-2], [-1], [1], [0], [1]]> : tensor<11x1xi32>
  %updates = util.unfoldable_constant dense<[40, -554, 29, 777, -304, -393, 583, 444, -493, 486, 497]> : tensor<11xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
    ^bb0(%current: tensor<i32>, %update: tensor<i32>):
      %sum = stablehlo.add %current, %update : tensor<i32>
      stablehlo.return %sum : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [0], scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>
  } : (tensor<1xi32>, tensor<11x1xi32>, tensor<11xi32>) -> tensor<1xi32>
  check.expect_eq_const(%result, dense<[611]> : tensor<1xi32>) : tensor<1xi32>
  return
}

func.func @scatter_oob_empty_dynamic_output() {
  %operand = flow.tensor.dynamic_constant dense<> : tensor<0xi32> -> tensor<?xi32>
  %indices = flow.tensor.dynamic_constant dense<[[0]]> : tensor<1x1xi32> -> tensor<?x1xi32>
  %updates = flow.tensor.dynamic_constant dense<[1]> : tensor<1xi32> -> tensor<?xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
    ^bb0(%current: tensor<i32>, %update: tensor<i32>):
      %sum = stablehlo.add %current, %update : tensor<i32>
      stablehlo.return %sum : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [0], scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>
  } : (tensor<?xi32>, tensor<?x1xi32>, tensor<?xi32>) -> tensor<?xi32>
  %expected = arith.constant dense<> : tensor<0xi32>
  %expected_dynamic = tensor.cast %expected : tensor<0xi32> to tensor<?xi32>
  check.expect_eq(%result, %expected_dynamic) : tensor<?xi32>
  return
}

// A runtime window extent equal to the operand's takes the whole-window
// scatter; the out-of-bounds row is dropped.
func.func @scatter_dynamic_whole_window() {
  %operand = flow.tensor.dynamic_constant dense<1> : tensor<5x3xi32> -> tensor<?x?xi32>
  %indices = flow.tensor.dynamic_constant dense<[[1], [-1], [3]]> : tensor<3x1xi32> -> tensor<?x1xi32>
  %updates = flow.tensor.dynamic_constant dense<[[1, 2, 3], [10, 20, 30], [4, 5, 6]]> : tensor<3x3xi32> -> tensor<?x?xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
    ^bb0(%current: tensor<i32>, %update: tensor<i32>):
      %sum = stablehlo.add %current, %update : tensor<i32>
      stablehlo.return %sum : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1], inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0], index_vector_dim = 1>
  } : (tensor<?x?xi32>, tensor<?x1xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  %static = tensor.cast %result : tensor<?x?xi32> to tensor<5x3xi32>
  check.expect_eq_const(%static, dense<[[1, 1, 1], [2, 3, 4], [1, 1, 1], [5, 6, 7], [1, 1, 1]]> : tensor<5x3xi32>) : tensor<5x3xi32>
  return
}

// A runtime window narrower than the operand takes the general scatter.
func.func @scatter_dynamic_narrow_window() {
  %operand = flow.tensor.dynamic_constant dense<1> : tensor<5x4xi32> -> tensor<?x?xi32>
  %indices = flow.tensor.dynamic_constant dense<[[4], [7], [0]]> : tensor<3x1xi32> -> tensor<?x1xi32>
  %updates = flow.tensor.dynamic_constant dense<[[1, 2, 3], [10, 20, 30], [4, 5, 6]]> : tensor<3x3xi32> -> tensor<?x?xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
    ^bb0(%current: tensor<i32>, %update: tensor<i32>):
      %sum = stablehlo.add %current, %update : tensor<i32>
      stablehlo.return %sum : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1], inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0], index_vector_dim = 1>
  } : (tensor<?x?xi32>, tensor<?x1xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  %static = tensor.cast %result : tensor<?x?xi32> to tensor<5x4xi32>
  check.expect_eq_const(%static, dense<[[5, 6, 7, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [2, 3, 4, 1]]> : tensor<5x4xi32>) : tensor<5x4xi32>
  return
}
