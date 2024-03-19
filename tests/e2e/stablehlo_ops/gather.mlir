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
