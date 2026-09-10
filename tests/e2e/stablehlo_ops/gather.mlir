// A runtime number of index vectors must bypass the static index-select reshape.
// Include both lower and upper out-of-bounds indices to check gather clamping.
func.func @gather_dynamic_index_count() {
  %a = flow.tensor.dynamic_constant dense<[[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]> : tensor<3x4xi32> -> tensor<?x4xi32>
  %i = flow.tensor.dynamic_constant dense<[[2], [-1], [3], [1]]> : tensor<4x1xi64> -> tensor<?x1xi64>
  %r = "stablehlo.gather"(%a, %i) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>, indices_are_sorted = false
  } : (tensor<?x4xi32>, tensor<?x1xi64>) -> tensor<?x4xi32>
  %expected = arith.constant dense<[[30, 31, 32, 33], [10, 11, 12, 13], [30, 31, 32, 33], [20, 21, 22, 23]]> : tensor<4x4xi32>
  %ed = tensor.cast %expected : tensor<4x4xi32> to tensor<?x4xi32>
  check.expect_eq(%r, %ed) : tensor<?x4xi32>
  return
}

func.func @gather_empty_index_count() {
  %a = util.unfoldable_constant dense<[[10, 11, 12, 13]]> : tensor<1x4xi32>
  %i = flow.tensor.dynamic_constant dense<> : tensor<0x1xi32> -> tensor<?x1xi32>
  %r = "stablehlo.gather"(%a, %i) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>, indices_are_sorted = false
  } : (tensor<1x4xi32>, tensor<?x1xi32>) -> tensor<?x4xi32>
  %dim = stablehlo.get_dimension_size %r, dim = 0 : (tensor<?x4xi32>) -> tensor<i32>
  check.expect_eq_const(%dim, dense<0> : tensor<i32>) : tensor<i32>
  return
}

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

// Keep slice sizes runtime values so this exercises dynamic_gather lowering.
func.func @dynamic_gather_leading_index_vector() {
  %a = util.unfoldable_constant dense<[[0, 1, 2, 3, 4, 5, 6, 7],
                                       [10, 11, 12, 13, 14, 15, 16, 17],
                                       [20, 21, 22, 23, 24, 25, 26, 27]]> : tensor<3x8xi32>
  %i = util.unfoldable_constant dense<[[[0, 7], [1, 6], [2, 5]]]> : tensor<1x3x2xi64>
  %s = util.unfoldable_constant dense<[1, 1]> : tensor<2xi64>
  %r = "stablehlo.dynamic_gather"(%a, %i, %s) {
    dimension_numbers = #stablehlo.gather<offset_dims = [], collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [1], start_index_map = [1], index_vector_dim = 0>,
    indices_are_sorted = false
  } : (tensor<3x8xi32>, tensor<1x3x2xi64>, tensor<2xi64>) -> tensor<3x2xi32>
  check.expect_eq_const(%r, dense<[[0, 7], [11, 16], [22, 25]]> : tensor<3x2xi32>) : tensor<3x2xi32>
  return
}
