func.func @foo() {
  %input = util.unfoldable_constant dense<[
    [[01, 02, 03, 04, 05]],
    [[06, 07, 08, 09, 10]],
    [[11, 12, 13, 14, 15]],
    [[16, 17, 18, 19, 20]],
    [[21, 22, 23, 24, 25]]]> : tensor<5x1x5xi32>
  %start_indices = util.unfoldable_constant dense<2> : tensor<i64>
  %res = "stablehlo.gather"(%input, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 0,
      offset_dims = [0, 1],
      start_index_map = [0],
    >,
    slice_sizes = array<i64: 1, 1, 5>
  } : (tensor<5x1x5xi32>, tensor<i64>) -> tensor<1x5xi32>
  check.expect_eq_const(%res, dense<[[11, 12, 13, 14, 15]]> : tensor<1x5xi32>) : tensor<1x5xi32>
  return
}

func.func @via_torch_index_select() {
  %input = util.unfoldable_constant dense<[
    [[01, 02, 03, 04, 05]],
    [[06, 07, 08, 09, 10]],
    [[11, 12, 13, 14, 15]],
    [[16, 17, 18, 19, 20]],
    [[21, 22, 23, 24, 25]]]> : tensor<5x1x5xi32>
  %start_indices = util.unfoldable_constant dense<2> : tensor<i64>
  %res = "stablehlo.gather"(%input, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 0,
      offset_dims = [0, 1],
      start_index_map = [0],
    >,
    slice_sizes = array<i64: 1, 1, 5>
  } : (tensor<5x1x5xi32>, tensor<i64>) -> tensor<1x5xi32>
  check.expect_eq_const(%res, dense<[[11, 12, 13, 14, 15]]> : tensor<1x5xi32>) : tensor<1x5xi32>
  return
}


func.func @general_but_just_index_select() {
  %operand = util.unfoldable_constant dense<[[
    [ 0,  1,  2,  3,  4,  5,  6,  7],
    [ 8,  9, 10, 11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29, 30, 31]]]> : tensor<1x4x8xi32>
  %start_indices = util.unfoldable_constant dense<[[
      [0, 1],
      [0, 2],
      [0, 3],
      [0, 0],
      [0, 0],
      [0, 1],
      [0, 2],
      [0, 3]]]> : tensor<1x8x2xi32>
  %result = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 1, 8>
  } : (tensor<1x4x8xi32>, tensor<1x8x2xi32>) -> tensor<1x8x8xi32>
  check.expect_eq_const(%result, dense<[[
         [ 8,  9, 10, 11, 12, 13, 14, 15],
         [16, 17, 18, 19, 20, 21, 22, 23],
         [24, 25, 26, 27, 28, 29, 30, 31],
         [ 0,  1,  2,  3,  4,  5,  6,  7],
         [ 0,  1,  2,  3,  4,  5,  6,  7],
         [ 8,  9, 10, 11, 12, 13, 14, 15],
         [16, 17, 18, 19, 20, 21, 22, 23],
         [24, 25, 26, 27, 28, 29, 30, 31]]]> : tensor<1x8x8xi32>) : tensor<1x8x8xi32>
  return
}

func.func @small_slices() {
  %operand = util.unfoldable_constant dense<[[
    [ 0,  1,  2,  3,  4,  5,  6,  7],
    [ 8,  9, 10, 11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29, 30, 31]]]> : tensor<1x4x8xi32>
  %start_indices = util.unfoldable_constant dense<[[
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 0]]]> : tensor<1x4x2xi32>
  %result = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 1, 3>
  } : (tensor<1x4x8xi32>, tensor<1x4x2xi32>) -> tensor<1x4x3xi32>
  check.expect_eq_const(%result, dense<[[
        [ 8,  9, 10],
        [16, 17, 18],
        [24, 25, 26],
        [ 0,  1,  2]]]> : tensor<1x4x3xi32>) : tensor<1x4x3xi32>
  return
}

func.func @nonstandard_offset_dims() {
  %operand = util.unfoldable_constant dense<[[
    [ 0,  1,  2,  3,  4,  5,  6,  7],
    [ 8,  9, 10, 11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29, 30, 31]]]> : tensor<1x4x8xi32>
  %start_indices = util.unfoldable_constant dense<[[
    [0, 1],
    [0, 2],
    [0, 2],
    [0, 0]]]> : tensor<1x4x2xi32>
  %result = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
      offset_dims = [1, 2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 2, 3>
  } : (tensor<1x4x8xi32>, tensor<1x4x2xi32>) -> tensor<1x2x3x4xi32>
  check.expect_eq_const(%result, dense<[[
      [[ 8, 16, 16,  0],
       [ 9, 17, 17,  1],
       [10, 18, 18,  2]],
      [[16, 24, 24,  8],
       [17, 25, 25,  9],
       [18, 26, 26, 10]]]]> : tensor<1x2x3x4xi32>) : tensor<1x2x3x4xi32>
  return
}

func.func @reordered_start_index() {
  %operand = util.unfoldable_constant dense<[[
    [[ 0,  1,  2,  3],
     [ 4,  5,  6,  7]],
    [[ 8,  9, 10, 11],
     [12, 13, 14, 15]],
    [[16, 17, 18, 19],
     [20, 21, 22, 23]]]]> : tensor<1x3x2x4xi32>
  %start_indices = util.unfoldable_constant dense<[
    [0, 1, 0, 0],
    [1, 0, 0, 0]]> : tensor<2x4xi32>
 %result = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 2],
      index_vector_dim = 1,
      offset_dims = [1, 2],
      start_index_map = [3, 2, 0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 2, 1, 3>
  } : (tensor<1x3x2x4xi32>, tensor<2x4xi32>) -> tensor<2x2x3xi32>

  check.expect_eq_const(%result, dense<[
    [[ 4,  5,  6],
     [12, 13, 14]],
    [[ 1,  2,  3],
     [ 9, 10, 11]]]> : tensor<2x2x3xi32>) : tensor<2x2x3xi32>
  return
}

func.func @clamp_index_select_indices() {
  %input = util.unfoldable_constant dense<[[10, 11], [20, 21], [30, 31]]> : tensor<3x2xi32>
  %indices = util.unfoldable_constant dense<[[-1], [0], [2], [3], [2147483647]]> : tensor<5x1xi32>
  %result = "stablehlo.gather"(%input, %indices) {
    dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], index_vector_dim = 1, offset_dims = [1], start_index_map = [0]>,
    slice_sizes = array<i64: 1, 2>
  } : (tensor<3x2xi32>, tensor<5x1xi32>) -> tensor<5x2xi32>
  check.expect_eq_const(%result, dense<[[10, 11], [10, 11], [30, 31], [30, 31], [30, 31]]> : tensor<5x2xi32>) : tensor<5x2xi32>
  return
}

func.func private @dynamic_index_select_gather(%input: tensor<?x2xi32>, %indices: tensor<4x1xi64>) -> tensor<4x2xi32> {
  %result = "stablehlo.gather"(%input, %indices) {
    dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], index_vector_dim = 1, offset_dims = [1], start_index_map = [0]>,
    slice_sizes = array<i64: 1, 2>
  } : (tensor<?x2xi32>, tensor<4x1xi64>) -> tensor<4x2xi32>
  return %result : tensor<4x2xi32>
}

func.func @clamp_dynamic_index_select_indices() {
  %input = util.unfoldable_constant dense<[[10, 11], [20, 21], [30, 31]]> : tensor<3x2xi32>
  %dynamic = tensor.cast %input : tensor<3x2xi32> to tensor<?x2xi32>
  %indices = util.unfoldable_constant dense<[[-9223372036854775808], [0], [2], [9223372036854775807]]> : tensor<4x1xi64>
  %result = call @dynamic_index_select_gather(%dynamic, %indices) : (tensor<?x2xi32>, tensor<4x1xi64>) -> tensor<4x2xi32>
  check.expect_eq_const(%result, dense<[[10, 11], [10, 11], [30, 31], [30, 31]]> : tensor<4x2xi32>) : tensor<4x2xi32>
  return
}

func.func @clamp_unsigned_index_select_indices() {
  %input = util.unfoldable_constant dense<[[10, 11], [20, 21], [30, 31]]> : tensor<3x2xi32>
  %bits = util.unfoldable_constant dense<[[0], [2], [-1]]> : tensor<3x1xi64>
  %indices = stablehlo.bitcast_convert %bits : (tensor<3x1xi64>) -> tensor<3x1xui64>
  %result = "stablehlo.gather"(%input, %indices) {
    dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], index_vector_dim = 1, offset_dims = [1], start_index_map = [0]>,
    slice_sizes = array<i64: 1, 2>
  } : (tensor<3x2xi32>, tensor<3x1xui64>) -> tensor<3x2xi32>
  check.expect_eq_const(%result, dense<[[10, 11], [30, 31], [30, 31]]> : tensor<3x2xi32>) : tensor<3x2xi32>
  return
}

func.func @clamp_narrow_index_select_indices() {
  %input = stablehlo.iota dim = 0 : tensor<256x1xi32>
  %indices = util.unfoldable_constant dense<[[-128], [0], [127]]> : tensor<3x1xi8>
  %result = "stablehlo.gather"(%input, %indices) {
    dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], index_vector_dim = 1, offset_dims = [1], start_index_map = [0]>,
    slice_sizes = array<i64: 1, 1>
  } : (tensor<256x1xi32>, tensor<3x1xi8>) -> tensor<3x1xi32>
  check.expect_eq_const(%result, dense<[[0], [0], [127]]> : tensor<3x1xi32>) : tensor<3x1xi32>
  return
}

func.func @dynamic_gather_runtime_slice_size_and_signed_clamp() {
  %operand = util.unfoldable_constant dense<[
      [0, 1, 2, 3, 4],
      [10, 11, 12, 13, 14],
      [20, 21, 22, 23, 24],
      [30, 31, 32, 33, 34],
      [40, 41, 42, 43, 44]]> : tensor<5x5xi32>
  %indices = util.unfoldable_constant dense<[[-1], [9]]> : tensor<2x1xi64>
  %slice_sizes = util.unfoldable_constant dense<[1, 3]> : tensor<2xi64>
  %result = "stablehlo.dynamic_gather"(%operand, %indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
    indices_are_sorted = false
  } : (tensor<5x5xi32>, tensor<2x1xi64>, tensor<2xi64>) -> tensor<2x?xi32>
  %expected = arith.constant dense<[[0, 1, 2], [40, 41, 42]]> : tensor<2x3xi32>
  %expected_dynamic = tensor.cast %expected : tensor<2x3xi32> to tensor<2x?xi32>
  check.expect_eq(%result, %expected_dynamic) : tensor<2x?xi32>
  return
}

func.func @dynamic_gather_unsigned_i64_max_clamps_high() {
  %operand = util.unfoldable_constant dense<[[10], [20], [30], [40], [50]]> : tensor<5x1xi32>
  %bits = util.unfoldable_constant dense<[[0], [-1]]> : tensor<2x1xi64>
  %indices = stablehlo.bitcast_convert %bits : (tensor<2x1xi64>) -> tensor<2x1xui64>
  %slice_sizes = util.unfoldable_constant dense<[1, 1]> : tensor<2xi64>
  %result = "stablehlo.dynamic_gather"(%operand, %indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
    indices_are_sorted = false
  } : (tensor<5x1xi32>, tensor<2x1xui64>, tensor<2xi64>) -> tensor<2x1xi32>
  check.expect_eq_const(%result, dense<[[10], [50]]> : tensor<2x1xi32>) : tensor<2x1xi32>
  return
}

func.func @dynamic_gather_narrow_i8_indices_on_large_operand() {
  %operand = stablehlo.iota dim = 0 : tensor<256x1xi32>
  %indices = util.unfoldable_constant dense<[[-128], [127]]> : tensor<2x1xi8>
  %slice_sizes = util.unfoldable_constant dense<[1, 1]> : tensor<2xi64>
  %result = "stablehlo.dynamic_gather"(%operand, %indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
    indices_are_sorted = false
  } : (tensor<256x1xi32>, tensor<2x1xi8>, tensor<2xi64>) -> tensor<2x1xi32>
  check.expect_eq_const(%result, dense<[[0], [127]]> : tensor<2x1xi32>) : tensor<2x1xi32>
  return
}

func.func @dynamic_gather_batching_runtime() {
  %operand = util.unfoldable_constant dense<[[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]> : tensor<2x5xi32>
  %indices = util.unfoldable_constant dense<[[[1], [3]], [[2], [4]]]> : tensor<2x2x1xi64>
  %slice_sizes = util.unfoldable_constant dense<[1, 1]> : tensor<2xi64>
  %result = "stablehlo.dynamic_gather"(%operand, %indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<offset_dims = [], collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>,
    indices_are_sorted = false
  } : (tensor<2x5xi32>, tensor<2x2x1xi64>, tensor<2xi64>) -> tensor<2x2xi32>
  check.expect_eq_const(%result, dense<[[1, 3], [12, 14]]> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

func.func @dynamic_gather_leading_index_vector_runtime() {
  %operand = util.unfoldable_constant dense<[[0, 1, 2, 3, 4, 5, 6, 7],
                                             [10, 11, 12, 13, 14, 15, 16, 17],
                                             [20, 21, 22, 23, 24, 25, 26, 27]]> : tensor<3x8xi32>
  %indices = util.unfoldable_constant dense<[[[0, 7], [1, 6], [2, 5]]]> : tensor<1x3x2xi64>
  %slice_sizes = util.unfoldable_constant dense<[1, 1]> : tensor<2xi64>
  %result = "stablehlo.dynamic_gather"(%operand, %indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<offset_dims = [], collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [1], start_index_map = [1], index_vector_dim = 0>,
    indices_are_sorted = false
  } : (tensor<3x8xi32>, tensor<1x3x2xi64>, tensor<2xi64>) -> tensor<3x2xi32>
  check.expect_eq_const(%result, dense<[[0, 7], [11, 16], [22, 25]]> : tensor<3x2xi32>) : tensor<3x2xi32>
  return
}


func.func private @wide_index_gather(%operand: tensor<?x4xi32>, %indices: tensor<2x1xi64>) -> tensor<2x4xi32> {
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<?x4xi32>, tensor<2x1xi64>) -> tensor<2x4xi32>
  return %result : tensor<2x4xi32>
}

func.func @wide_index_clamps_before_target_index_cast() {
  %operand = util.unfoldable_constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]> : tensor<3x4xi32>
  %dynamic_operand = tensor.cast %operand : tensor<3x4xi32> to tensor<?x4xi32>
  %indices = util.unfoldable_constant dense<[[4294967296], [-4294967296]]> : tensor<2x1xi64>
  %result = call @wide_index_gather(%dynamic_operand, %indices) : (tensor<?x4xi32>, tensor<2x1xi64>) -> tensor<2x4xi32>
  check.expect_eq_const(%result, dense<[[8, 9, 10, 11], [0, 1, 2, 3]]> : tensor<2x4xi32>) : tensor<2x4xi32>
  return
}

func.func @static_wide_index_select_clamps() {
  %operand = util.unfoldable_constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]> : tensor<3x4xi32>
  %indices = util.unfoldable_constant dense<[[4294967296], [-4294967296]]> : tensor<2x1xi64>
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<3x4xi32>, tensor<2x1xi64>) -> tensor<2x4xi32>
  check.expect_eq_const(%result, dense<[[8, 9, 10, 11], [0, 1, 2, 3]]> : tensor<2x4xi32>) : tensor<2x4xi32>
  return
}

func.func @static_unsigned_index_select_clamps() {
  %operand = util.unfoldable_constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]> : tensor<3x4xi32>
  %bits = util.unfoldable_constant dense<[[-1], [0]]> : tensor<2x1xi64>
  %indices = stablehlo.convert %bits : (tensor<2x1xi64>) -> tensor<2x1xui64>
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>
  } : (tensor<3x4xi32>, tensor<2x1xui64>) -> tensor<2x4xi32>
  check.expect_eq_const(%result, dense<[[8, 9, 10, 11], [0, 1, 2, 3]]> : tensor<2x4xi32>) : tensor<2x4xi32>
  return
}
