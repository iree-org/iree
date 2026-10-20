// General scatter preserves per-element bounds and a coupled reducer schedule.
func.func @scatter_overlap_dynamic_vector_and_wide_starts() {
  %input = flow.tensor.dynamic_constant dense<0> : tensor<5xi32> -> tensor<?xi32>
  %indices = flow.tensor.dynamic_constant dense<[[-1], [3], [-9223372036854775808], [9223372036854775807], [4294967297]]> : tensor<5x1xi64> -> tensor<?x?xi64>
  %updates = flow.tensor.dynamic_constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]> : tensor<5x3xi32> -> tensor<?x?xi32>
  %result = "stablehlo.scatter"(%input, %indices, %updates) ({
  ^bb0(%current: tensor<i32>, %update: tensor<i32>):
    %sum = stablehlo.add %current, %update : tensor<i32>
    stablehlo.return %sum : tensor<i32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : (tensor<?xi32>, tensor<?x?xi64>, tensor<?x?xi32>) -> tensor<?xi32>
  %expected = arith.constant dense<[2, 3, 0, 4, 5]> : tensor<5xi32>
  %dynamic = tensor.cast %expected : tensor<5xi32> to tensor<?xi32>
  check.expect_eq(%result, %dynamic) : tensor<?xi32>
  return
}

// Identical duplicate updates make the result independent of schedule order.
func.func @scatter_coupled_current_values() {
  %input0 = flow.tensor.dynamic_constant dense<1> : tensor<3xi32> -> tensor<?xi32>
  %input1 = flow.tensor.dynamic_constant dense<10> : tensor<3xi32> -> tensor<?xi32>
  %indices = flow.tensor.dynamic_constant dense<1> : tensor<2x1xi64> -> tensor<?x1xi64>
  %updates0 = flow.tensor.dynamic_constant dense<2> : tensor<2xi32> -> tensor<?xi32>
  %updates1 = flow.tensor.dynamic_constant dense<3> : tensor<2xi32> -> tensor<?xi32>
  %result:2 = "stablehlo.scatter"(%input0, %input1, %indices, %updates0, %updates1) ({
  ^bb0(%current0: tensor<i32>, %current1: tensor<i32>, %update0: tensor<i32>, %update1: tensor<i32>):
    %sum = stablehlo.add %current0, %current1 : tensor<i32>
    %next0 = stablehlo.add %sum, %update0 : tensor<i32>
    %next1 = stablehlo.add %sum, %update1 : tensor<i32>
    stablehlo.return %next0, %next1 : tensor<i32>, tensor<i32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : (tensor<?xi32>, tensor<?xi32>, tensor<?x1xi64>, tensor<?xi32>, tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  %expected0 = arith.constant dense<[1, 29, 1]> : tensor<3xi32>
  %expected1 = arith.constant dense<[10, 30, 10]> : tensor<3xi32>
  %dynamic0 = tensor.cast %expected0 : tensor<3xi32> to tensor<?xi32>
  %dynamic1 = tensor.cast %expected1 : tensor<3xi32> to tensor<?xi32>
  check.expect_eq(%result#0, %dynamic0) : tensor<?xi32>
  check.expect_eq(%result#1, %dynamic1) : tensor<?xi32>
  return
}

func.func @scatter_nonleading_batch_axis_overlap() {
  %input = flow.tensor.dynamic_constant dense<0> : tensor<2x4xi32> -> tensor<?x4xi32>
  %indices = flow.tensor.dynamic_constant dense<[[[-1], [3]]]> : tensor<1x2x1xi64> -> tensor<1x?x1xi64>
  %updates = flow.tensor.dynamic_constant dense<[[[1, 2], [3, 4]]]> : tensor<1x2x2xi32> -> tensor<1x?x2xi32>
  %result = "stablehlo.scatter"(%input, %indices, %updates) ({
  ^bb0(%current: tensor<i32>, %update: tensor<i32>):
    %sum = stablehlo.add %current, %update : tensor<i32>
    stablehlo.return %sum : tensor<i32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [2], input_batching_dims = [0], scatter_indices_batching_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, indices_are_sorted = false, unique_indices = false} : (tensor<?x4xi32>, tensor<1x?x1xi64>, tensor<1x?x2xi32>) -> tensor<?x4xi32>
  %expected = arith.constant dense<[[2, 0, 0, 0], [0, 0, 0, 3]]> : tensor<2x4xi32>
  %dynamic = tensor.cast %expected : tensor<2x4xi32> to tensor<?x4xi32>
  check.expect_eq(%result, %dynamic) : tensor<?x4xi32>
  return
}

// An overwrite window starting before zero drops its first element.
func.func @scatter_partial_overwrite_window() {
  %input = util.unfoldable_constant dense<0> : tensor<4xi32>
  %indices = stablehlo.constant dense<-1> : tensor<1xi64>
  %updates = util.unfoldable_constant dense<[1, 2]> : tensor<2xi32>
  %result = "stablehlo.scatter"(%input, %indices, %updates) ({
  ^bb0(%current: tensor<i32>, %update: tensor<i32>):
    stablehlo.return %update : tensor<i32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 0>, indices_are_sorted = false, unique_indices = false} : (tensor<4xi32>, tensor<1xi64>, tensor<2xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[2, 0, 0, 0]> : tensor<4xi32>) : tensor<4xi32>
  return
}

// Unsigned starts remain nonnegative even when their high bit is set.
func.func @scatter_unsigned_wide_starts() {
  %input = util.unfoldable_constant dense<0> : tensor<4xi32>
  %raw_indices = flow.tensor.dynamic_constant dense<[[-1], [-9223372036854775808], [1]]> : tensor<3x1xi64> -> tensor<?x1xi64>
  %indices = stablehlo.bitcast_convert %raw_indices : (tensor<?x1xi64>) -> tensor<?x1xui64>
  %updates = flow.tensor.dynamic_constant dense<[[7, 8], [9, 10], [2, 3]]> : tensor<3x2xi32> -> tensor<?x2xi32>
  %result = "stablehlo.scatter"(%input, %indices, %updates) ({
  ^bb0(%current: tensor<i32>, %update: tensor<i32>):
    %sum = stablehlo.add %current, %update : tensor<i32>
    stablehlo.return %sum : tensor<i32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = true} : (tensor<4xi32>, tensor<?x1xui64>, tensor<?x2xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[0, 2, 3, 0]> : tensor<4xi32>) : tensor<4xi32>
  return
}

func.func @scatter_promoted_variadic_elements() {
  %input0 = util.unfoldable_constant dense<1.5> : tensor<3xf16>
  %input1 = util.unfoldable_constant dense<10> : tensor<3xi8>
  %indices = util.unfoldable_constant dense<1> : tensor<2x1xi64>
  %updates0 = util.unfoldable_constant dense<2.0> : tensor<2xf16>
  %updates1 = util.unfoldable_constant dense<3> : tensor<2xi8>
  %result:2 = "stablehlo.scatter"(%input0, %input1, %indices, %updates0, %updates1) ({
  ^bb0(%current0: tensor<f32>, %current1: tensor<i32>, %update0: tensor<f32>, %update1: tensor<i32>):
    %current1_float = stablehlo.convert %current1 : (tensor<i32>) -> tensor<f32>
    %sum = stablehlo.add %current0, %current1_float : tensor<f32>
    %next0 = stablehlo.add %sum, %update0 : tensor<f32>
    %next1 = stablehlo.add %current1, %update1 : tensor<i32>
    stablehlo.return %next0, %next1 : tensor<f32>, tensor<i32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : (tensor<3xf16>, tensor<3xi8>, tensor<2x1xi64>, tensor<2xf16>, tensor<2xi8>) -> (tensor<3xf32>, tensor<3xi32>)
  check.expect_eq_const(%result#0, dense<[1.5, 28.5, 1.5]> : tensor<3xf32>) : tensor<3xf32>
  check.expect_eq_const(%result#1, dense<[10, 16, 10]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func.func @scatter_unsigned_scalar_zero_index_depth() {
  %raw_input = util.unfoldable_constant dense<7> : tensor<i32>
  %input = stablehlo.convert %raw_input : (tensor<i32>) -> tensor<ui32>
  %indices = util.unfoldable_constant dense<> : tensor<0xi64>
  %raw_update = util.unfoldable_constant dense<4> : tensor<i32>
  %update = stablehlo.convert %raw_update : (tensor<i32>) -> tensor<ui32>
  %result = "stablehlo.scatter"(%input, %indices, %update) ({
  ^bb0(%current: tensor<ui32>, %new: tensor<ui32>):
    %sum = stablehlo.add %current, %new : tensor<ui32>
    stablehlo.return %sum : tensor<ui32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<index_vector_dim = 0>, indices_are_sorted = false, unique_indices = false} : (tensor<ui32>, tensor<0xi64>, tensor<ui32>) -> tensor<ui32>
  %signed = stablehlo.convert %result : (tensor<ui32>) -> tensor<i32>
  check.expect_eq_const(%signed, dense<11> : tensor<i32>) : tensor<i32>
  return
}

func.func @scatter_empty_update_batch() {
  %input = util.unfoldable_constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  %indices = flow.tensor.dynamic_constant dense<> : tensor<0x1xi64> -> tensor<?x1xi64>
  %updates = flow.tensor.dynamic_constant dense<> : tensor<0x2xi32> -> tensor<?x2xi32>
  %result = "stablehlo.scatter"(%input, %indices, %updates) ({
  ^bb0(%current: tensor<i32>, %update: tensor<i32>):
    %sum = stablehlo.add %current, %update : tensor<i32>
    stablehlo.return %sum : tensor<i32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : (tensor<5xi32>, tensor<?x1xi64>, tensor<?x2xi32>) -> tensor<5xi32>
  check.expect_eq_const(%result, dense<[1, 2, 3, 4, 5]> : tensor<5xi32>) : tensor<5xi32>
  return
}

func.func @scatter_permuted_window_and_index_axes() {
  %input = util.unfoldable_constant dense<0> : tensor<3x4xi32>
  %indices = util.unfoldable_constant dense<[[1], [0]]> : tensor<2x1xi64>
  %updates = util.unfoldable_constant dense<[[[1, 2]], [[3, 4]]]> : tensor<2x1x2xi32>
  %result = "stablehlo.scatter"(%input, %indices, %updates) ({
  ^bb0(%current: tensor<i32>, %update: tensor<i32>):
    %sum = stablehlo.add %current, %update : tensor<i32>
    stablehlo.return %sum : tensor<i32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], scatter_dims_to_operand_dims = [1, 0], index_vector_dim = 0>, indices_are_sorted = false, unique_indices = false} : (tensor<3x4xi32>, tensor<2x1xi64>, tensor<2x1x2xi32>) -> tensor<3x4xi32>
  check.expect_eq_const(%result, dense<[[0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]> : tensor<3x4xi32>) : tensor<3x4xi32>
  return
}
