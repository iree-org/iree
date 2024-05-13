func.func @i4_to_f32_1d() {
  %input = util.unfoldable_constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi4>
  %0 = tensor.empty() : tensor<8xf32>
  %res = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
    ins(%input : tensor<8xi4>) outs(%0 : tensor<8xf32>) {
  ^bb0(%in: i4, %out: f32):
    %2 = arith.extui %in : i4 to i32
    %3 = arith.uitofp %2 : i32 to f32
    linalg.yield %3 : f32
  } -> tensor<8xf32>
  check.expect_eq_const(%res, dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]> : tensor<8xf32>) : tensor<8xf32>
  return
}

func.func @i4_to_f32_3d() {
  %cst = util.unfoldable_constant dense<[
    [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15],
    [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15],
    [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15],
    [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]
  ]> : tensor<8x8xi4>
  %expanded_4 = tensor.expand_shape %cst [[0], [1, 2]] output_shape [8, 4, 2]: tensor<8x8xi4> into tensor<8x4x2xi4>
  %0 = tensor.empty() : tensor<8x4x2xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_4 : tensor<8x4x2xi4>) outs(%0 : tensor<8x4x2xf32>) {
  ^bb0(%in: i4, %out: f32):
    %6 = arith.extui %in : i4 to i32
    %7 = arith.uitofp %6 : i32 to f32
    linalg.yield %7 : f32
  } -> tensor<8x4x2xf32>
  check.expect_almost_eq_const(%5, dense<[
    [[0., 1.], [2., 3.], [4., 5.], [6., 7.]], [[8., 9.], [10., 11.], [12., 13.], [14., 15.]],
    [[0., 1.], [2., 3.], [4., 5.], [6., 7.]], [[8., 9.], [10., 11.], [12., 13.], [14., 15.]],
    [[0., 1.], [2., 3.], [4., 5.], [6., 7.]], [[8., 9.], [10., 11.], [12., 13.], [14., 15.]],
    [[0., 1.], [2., 3.], [4., 5.], [6., 7.]], [[8., 9.], [10., 11.], [12., 13.], [14., 15.]]
  ]> : tensor<8x4x2xf32>) : tensor<8x4x2xf32>
  return
}

func.func @i2_to_f32_1d() {
  %input = util.unfoldable_constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi2>
  %0 = tensor.empty() : tensor<8xf32>
  %res = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
    ins(%input : tensor<8xi2>) outs(%0 : tensor<8xf32>) {
  ^bb0(%in: i2, %out: f32):
    %2 = arith.extui %in : i2 to i32
    %3 = arith.uitofp %2 : i32 to f32
    linalg.yield %3 : f32
  } -> tensor<8xf32>
  check.expect_eq_const(%res, dense<[0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0]> : tensor<8xf32>) : tensor<8xf32>
  return
}

func.func @i2_to_f32_3d() {
  %cst = util.unfoldable_constant dense<[
    [0, 1, 2, 3, 0, 1, 2, 3], [3, 2, 1, 0, 2, 1, 0, 3],
    [0, 1, 2, 3, 0, 1, 2, 3], [3, 2, 1, 0, 2, 1, 0, 3],
    [0, 1, 2, 3, 0, 1, 2, 3], [3, 2, 1, 0, 2, 1, 0, 3],
    [0, 1, 2, 3, 0, 1, 2, 3], [3, 2, 1, 0, 2, 1, 0, 3]
  ]> : tensor<8x8xi2>
  %expanded_4 = tensor.expand_shape %cst [[0], [1, 2]] output_shape [8, 4, 2]: tensor<8x8xi2> into tensor<8x4x2xi2>
  %0 = tensor.empty() : tensor<8x4x2xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_4 : tensor<8x4x2xi2>) outs(%0 : tensor<8x4x2xf32>) {
  ^bb0(%in: i2, %out: f32):
    %6 = arith.extui %in : i2 to i32
    %7 = arith.uitofp %6 : i32 to f32
    linalg.yield %7 : f32
  } -> tensor<8x4x2xf32>
  check.expect_almost_eq_const(%5, dense<[
    [[0., 1.], [2., 3.], [0., 1.], [2., 3.]], [[3., 2.], [1., 0.], [2., 1.], [0., 3.]],
    [[0., 1.], [2., 3.], [0., 1.], [2., 3.]], [[3., 2.], [1., 0.], [2., 1.], [0., 3.]],
    [[0., 1.], [2., 3.], [0., 1.], [2., 3.]], [[3., 2.], [1., 0.], [2., 1.], [0., 3.]],
    [[0., 1.], [2., 3.], [0., 1.], [2., 3.]], [[3., 2.], [1., 0.], [2., 1.], [0., 3.]]
  ]> : tensor<8x4x2xf32>) : tensor<8x4x2xf32>
  return
}
