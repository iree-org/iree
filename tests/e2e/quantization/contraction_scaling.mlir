// Runs with the integer rewrite enabled and disabled.
// Inputs stay unfolded so these checks execute the compiled contractions.

// Both correction products and the cross term vary by block and output axis.
func.func @blockwise_zero_point_corrections() {
  %aq = util.unfoldable_constant dense<[[[2, 3], [4, 6]], [[5, 8], [9, 10]]]> : tensor<2x2x2xi8>
  %bq = util.unfoldable_constant dense<[[[2, 4, 6], [3, 5, 8]], [[6, 7, 9], [7, 9, 10]]]> : tensor<2x2x3xi8>
  %sa = util.unfoldable_constant dense<[[1.0, 0.5], [2.0, 1.0]]> : tensor<2x2xf32>
  %sb = util.unfoldable_constant dense<[[1.0, 2.0, 1.0], [0.5, 1.0, 2.0]]> : tensor<2x3xf32>
  %za = util.unfoldable_constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi8>
  %zb = util.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi8>
  %ai = tensor.empty() : tensor<2x2x2xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(m, g, k) -> (m, g, k)>,
                        affine_map<(m, g, k) -> (m, g)>,
                        affine_map<(m, g, k) -> (m, g)>,
                        affine_map<(m, g, k) -> (m, g, k)>]}
      ins(%aq, %sa, %za : tensor<2x2x2xi8>, tensor<2x2xf32>, tensor<2x2xi8>)
      outs(%ai : tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
  %bi = tensor.empty() : tensor<2x2x3xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(g, k, n) -> (g, k, n)>,
                        affine_map<(g, k, n) -> (g, n)>,
                        affine_map<(g, k, n) -> (g, n)>,
                        affine_map<(g, k, n) -> (g, k, n)>]}
      ins(%bq, %sb, %zb : tensor<2x2x3xi8>, tensor<2x3xf32>, tensor<2x3xi8>)
      outs(%bi : tensor<2x2x3xf32>) -> tensor<2x2x3xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x3xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x3xf32>) -> tensor<2x3xf32>
  %result = linalg.contract
      indexing_maps = [affine_map<(m, n, g, k) -> (m, g, k)>,
                       affine_map<(m, n, g, k) -> (g, k, n)>,
                       affine_map<(m, n, g, k) -> (m, n)>]
      ins(%a, %b : tensor<2x2x2xf32>, tensor<2x2x3xf32>)
      outs(%init : tensor<2x3xf32>) -> tensor<2x3xf32>
  check.expect_eq_const(%result, dense<[[9.0, 26.0, 35.0], [38.0, 110.0, 140.0]]> : tensor<2x3xf32>) : tensor<2x3xf32>
  return
}

// Nonuniform image values distinguish the strided/dilated windows. Each output
// channel has a different zero point and scale but the same dequantized filter.
func.func @convolution_zero_point_corrections() {
  %aq = util.unfoldable_constant dense<[[[[1], [2], [3], [4], [5]], [[6], [7], [8], [9], [10]], [[11], [12], [13], [14], [15]], [[16], [17], [18], [19], [20]], [[21], [22], [23], [24], [25]]]]> : tensor<1x5x5x1xi8>
  %bq = util.unfoldable_constant dense<[[[[7, 9]], [[7, 9]]], [[[7, 9]], [[7, 9]]]]> : tensor<2x2x1x2xi8>
  %sa = util.unfoldable_constant 0.5 : f32
  %sb = util.unfoldable_constant dense<[1.0, 0.5]> : tensor<2xf32>
  %za = util.unfoldable_constant 3 : i8
  %zb = util.unfoldable_constant dense<[6, 7]> : tensor<2xi8>
  %ai = tensor.empty() : tensor<1x5x5x1xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(n, h, w, c) -> (n, h, w, c)>,
                        affine_map<(n, h, w, c) -> ()>,
                        affine_map<(n, h, w, c) -> ()>,
                        affine_map<(n, h, w, c) -> (n, h, w, c)>]}
      ins(%aq, %sa, %za : tensor<1x5x5x1xi8>, f32, i8)
      outs(%ai : tensor<1x5x5x1xf32>) -> tensor<1x5x5x1xf32>
  %bi = tensor.empty() : tensor<2x2x1x2xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(h, w, c, f) -> (h, w, c, f)>,
                        affine_map<(h, w, c, f) -> (f)>,
                        affine_map<(h, w, c, f) -> (f)>,
                        affine_map<(h, w, c, f) -> (h, w, c, f)>]}
      ins(%bq, %sb, %zb : tensor<2x2x1x2xi8>, tensor<2xf32>, tensor<2xi8>)
      outs(%bi : tensor<2x2x1x2xf32>) -> tensor<2x2x1x2xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x2x2x2xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<2> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%a, %b : tensor<1x5x5x1xf32>, tensor<2x2x1x2xf32>)
      outs(%init : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  check.expect_eq_const(%result, dense<[[[[8.0, 8.0], [12.0, 12.0]], [[28.0, 28.0], [32.0, 32.0]]]]> : tensor<1x2x2x2xf32>) : tensor<1x2x2x2xf32>
  return
}
