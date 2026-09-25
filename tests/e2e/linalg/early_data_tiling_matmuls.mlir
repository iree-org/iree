// Numerical checks for data-tiling layouts materialized before dispatch
// creation. The shapes are deliberately not multiples of the CPU tile sizes,
// so the packed operands and the fused epilogues run on padded tiles.

func.func @matmul_relu_odd_shapes() {
  %lhs = util.unfoldable_constant dense<[[2.0, 1.0, 0.0, -2.0, -1.0, -3.0, -3.0], [-3.0, -2.0, 2.0, 1.0, 3.0, 0.0, 1.0], [3.0, 2.0, 1.0, 0.0, 0.0, 3.0, -2.0], [2.0, 1.0, -3.0, -1.0, 3.0, 0.0, -3.0], [2.0, 2.0, 2.0, -2.0, -3.0, 3.0, -3.0]]> : tensor<5x7xf32>
  %rhs = util.unfoldable_constant dense<[[0.0, -3.0, -1.0], [0.0, -1.0, -1.0], [-3.0, -3.0, -3.0], [-3.0, 1.0, 0.0], [1.0, -2.0, 1.0], [2.0, -1.0, 0.0], [3.0, 2.0, 3.0]]> : tensor<7x3xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<5x3xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<5x3xf32>) -> tensor<5x3xf32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<5x7xf32>, tensor<7x3xf32>) outs(%init : tensor<5x3xf32>) -> tensor<5x3xf32>
  %relu = linalg.generic {indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m, n)>], iterator_types = ["parallel", "parallel"]} ins(%mm : tensor<5x3xf32>) outs(%empty : tensor<5x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<5x3xf32>
  check.expect_almost_eq_const(%relu, dense<[[0.0, 0.0, 0.0], [0.0, 2.0, 5.0], [0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 0.0, 0.0]]> : tensor<5x3xf32>) : tensor<5x3xf32>
  return
}

func.func @matmul_bias_relu_dynamic_m() {
  %lhs = flow.tensor.dynamic_constant dense<[[-1.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0], [-1.0, 3.0, -3.0, 1.0, 2.0, 2.0, 0.0], [-1.0, -1.0, -1.0, 0.0, 2.0, 3.0, -3.0], [3.0, 0.0, -1.0, 1.0, 1.0, -2.0, -1.0], [2.0, 1.0, 0.0, -1.0, 2.0, -1.0, -1.0]]> : tensor<5x7xf32> -> tensor<?x7xf32>
  %rhs = util.unfoldable_constant dense<[[3.0, -2.0, -2.0], [1.0, 1.0, -3.0], [-3.0, -1.0, 2.0], [-1.0, 2.0, -1.0], [-2.0, 2.0, 3.0], [-3.0, -3.0, 1.0], [-1.0, 1.0, -2.0]]> : tensor<7x3xf32>
  %bias = util.unfoldable_constant dense<[3.0, 0.0, 3.0]> : tensor<3xf32>
  %c0 = arith.constant 0 : index
  %m = tensor.dim %lhs, %c0 : tensor<?x7xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty(%m) : tensor<?x3xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x3xf32>) -> tensor<?x3xf32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<?x7xf32>, tensor<7x3xf32>) outs(%init : tensor<?x3xf32>) -> tensor<?x3xf32>
  %result = linalg.generic {indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (n)>, affine_map<(m, n) -> (m, n)>], iterator_types = ["parallel", "parallel"]} ins(%mm, %bias : tensor<?x3xf32>, tensor<3xf32>) outs(%empty : tensor<?x3xf32>) {
  ^bb0(%in: f32, %b: f32, %out: f32):
    %sum = arith.addf %in, %b : f32
    %max = arith.maximumf %sum, %zero : f32
    linalg.yield %max : f32
  } -> tensor<?x3xf32>
  %cast = tensor.cast %result : tensor<?x3xf32> to tensor<5x3xf32>
  check.expect_almost_eq_const(%cast, dense<[[0.0, 4.0, 12.0], [1.0, 8.0, 0.0], [0.0, 0.0, 21.0], [19.0, 4.0, 0.0], [11.0, 1.0, 4.0]]> : tensor<5x3xf32>) : tensor<5x3xf32>
  return
}

func.func @batch_matmul_relu() {
  %lhs = util.unfoldable_constant dense<[[[2.0, 1.0, -2.0, 2.0, -3.0], [0.0, -1.0, 3.0, -2.0, 3.0], [-3.0, 1.0, 1.0, 3.0, -1.0]], [[3.0, 1.0, 3.0, -2.0, 2.0], [3.0, -3.0, -1.0, 1.0, -3.0], [0.0, 1.0, 2.0, 3.0, -1.0]]]> : tensor<2x3x5xf32>
  %rhs = util.unfoldable_constant dense<[[[0.0, 0.0], [3.0, -2.0], [0.0, -3.0], [-1.0, 3.0], [1.0, -1.0]], [[3.0, 1.0], [3.0, -3.0], [0.0, 2.0], [2.0, -1.0], [0.0, -1.0]]]> : tensor<2x5x2xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x3x2xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x3x2xf32>) -> tensor<2x3x2xf32>
  %mm = linalg.batch_matmul ins(%lhs, %rhs : tensor<2x3x5xf32>, tensor<2x5x2xf32>) outs(%init : tensor<2x3x2xf32>) -> tensor<2x3x2xf32>
  %relu = linalg.generic {indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>, affine_map<(b, m, n) -> (b, m, n)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%mm : tensor<2x3x2xf32>) outs(%empty : tensor<2x3x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<2x3x2xf32>
  check.expect_almost_eq_const(%relu, dense<[[[0.0, 13.0], [2.0, 0.0], [0.0, 5.0]], [[8.0, 6.0], [2.0, 12.0], [9.0, 0.0]]]> : tensor<2x3x2xf32>) : tensor<2x3x2xf32>
  return
}

func.func @multi_m_contraction_relu() {
  %lhs = util.unfoldable_constant dense<[[[0.0, -2.0, 2.0, -3.0], [-1.0, -2.0, 2.0, 2.0], [1.0, 3.0, 3.0, -2.0]], [[-3.0, -3.0, 2.0, 3.0], [3.0, 1.0, 3.0, 3.0], [-3.0, -3.0, 3.0, -3.0]]]> : tensor<2x3x4xf32>
  %rhs = util.unfoldable_constant dense<[[3.0, 2.0, 3.0, -1.0, -2.0], [0.0, 3.0, -1.0, 3.0, -1.0], [2.0, -2.0, 0.0, -1.0, -2.0], [3.0, 2.0, -3.0, 3.0, 3.0]]> : tensor<4x5xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x3x5xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x3x5xf32>) -> tensor<2x3x5xf32>
  %mm = linalg.generic {indexing_maps = [affine_map<(m0, m1, n, k) -> (m0, m1, k)>, affine_map<(m0, m1, n, k) -> (k, n)>, affine_map<(m0, m1, n, k) -> (m0, m1, n)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%lhs, %rhs : tensor<2x3x4xf32>, tensor<4x5xf32>) outs(%init : tensor<2x3x5xf32>) {
  ^bb0(%l: f32, %r: f32, %acc: f32):
    %mul = arith.mulf %l, %r : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x3x5xf32>
  %relu = linalg.generic {indexing_maps = [affine_map<(a, b, c) -> (a, b, c)>, affine_map<(a, b, c) -> (a, b, c)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%mm : tensor<2x3x5xf32>) outs(%empty : tensor<2x3x5xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<2x3x5xf32>
  check.expect_almost_eq_const(%relu, dense<[[[0.0, 0.0, 11.0, 0.0, 0.0], [7.0, 0.0, 0.0, 0.0, 6.0], [3.0, 1.0, 6.0, 0.0, 0.0]], [[4.0, 0.0, 0.0, 1.0, 14.0], [24.0, 9.0, 0.0, 6.0, 0.0], [0.0, 0.0, 3.0, 0.0, 0.0]]]> : tensor<2x3x5xf32>) : tensor<2x3x5xf32>
  return
}

func.func @dequantized_rhs_matmul() {
  %lhs = util.unfoldable_constant dense<[[-2.0, -1.0, 0.0, 1.0, 0.0, -2.0, 3.0], [1.0, -3.0, 2.0, 2.0, -2.0, 1.0, 0.0], [-3.0, 3.0, 2.0, -1.0, -3.0, -3.0, 2.0], [-2.0, 0.0, 3.0, 3.0, -2.0, -3.0, 0.0], [2.0, 1.0, -3.0, 1.0, -1.0, -2.0, 0.0]]> : tensor<5x7xf32>
  %weights = util.unfoldable_constant dense<[[5, 7, -6], [0, 4, -4], [-4, -5, -5], [6, -5, -5], [-6, -7, 4], [-4, 4, 1], [5, 0, 4]]> : tensor<7x3xi8>
  %scale = arith.constant 0.5 : f32
  %wide_empty = tensor.empty() : tensor<7x3xf32>
  %rhs = linalg.generic {indexing_maps = [affine_map<(k, n) -> (k, n)>, affine_map<(k, n) -> (k, n)>], iterator_types = ["parallel", "parallel"]} ins(%weights : tensor<7x3xi8>) outs(%wide_empty : tensor<7x3xf32>) {
  ^bb0(%in: i8, %out: f32):
    %wide = arith.sitofp %in : i8 to f32
    %scaled = arith.mulf %wide, %scale : f32
    linalg.yield %scaled : f32
  } -> tensor<7x3xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<5x3xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<5x3xf32>) -> tensor<5x3xf32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<5x7xf32>, tensor<7x3xf32>) outs(%init : tensor<5x3xf32>) -> tensor<5x3xf32>
  check.expect_almost_eq_const(%mm, dense<[[9.5, -15.5, 10.5], [8.5, -3.5, -10.5], [5.5, -2.5, -3.0], [10.0, -21.0, -14.5], [21.0, 13.5, -6.0]]> : tensor<5x3xf32>) : tensor<5x3xf32>
  return
}

func.func @i8_matmul_odd_shapes() {
  %lhs = util.unfoldable_constant dense<[[4, -8, 0, -1, -4, -1, -2], [-1, 5, 5, 2, 3, 7, 2], [-3, -7, 0, -5, 1, -8, 5], [7, -6, 5, -2, -8, 6, 7], [-8, 1, 5, 4, -2, 5, 5]]> : tensor<5x7xi8>
  %rhs = util.unfoldable_constant dense<[[-7, -8, -7], [-3, -7, -7], [-4, 2, 0], [-4, 7, 3], [2, 7, 4], [-6, -8, 5], [-2, -8, -1]]> : tensor<7x3xi8>
  %zero = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<5x3xi32>
  %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<5x3xi32>) -> tensor<5x3xi32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<5x7xi8>, tensor<7x3xi8>) outs(%init : tensor<5x3xi32>) -> tensor<5x3xi32>
  check.expect_eq_const(%mm, dense<[[2, 13, 6], [-76, -54, 23], [102, 69, 14], [-109, -178, -22], [-27, 1, 73]]> : tensor<5x3xi32>) : tensor<5x3xi32>
  return
}
