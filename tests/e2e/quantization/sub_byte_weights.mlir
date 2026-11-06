// Contractions over constant weights whose quantization range fits 2 or 4
// bits. With --iree-global-opt-enable-sub-byte-quantized-storage the weights
// are packed into i2 or i4 storage; without it they stay i8. Both must give
// the same results. Only constants are packed, so the weights are inline;
// the activations are unfolded so the contractions run at runtime. Scales
// are powers of two, so the expected values are exact.

// Signed 4-bit weights. Constant evaluation sums the packed weights for the
// activation zero point correction.
func.func @signed_i4_weights_asymmetric_activations() {
  %xq = util.unfoldable_constant dense<[[-6, 16, -12, -2, -15, 2, 11, 10, -19, -15, 2, 9, -13, -6, -20, 11], [-1, -18, -10, -10, -5, -18, -5, 18, 18, 6, 15, 2, 20, -5, 1, 17]]> : tensor<2x16xi8>
  %weight = arith.constant dense<[[6, -2, 4, 5, 6, 5, -1, 5, 7, 2, 4, -3, -5, -7, 4, -6], [-5, -6, 4, 5, 7, -5, -1, 4, -3, 0, -2, 7, -6, 0, 2, -3], [4, 7, 7, -6, 3, -6, -8, -7, 3, -2, 2, 0, -5, 7, -5, -7], [-6, 1, -1, -6, 7, 1, -3, -7, 5, -7, 6, 1, -8, -2, 1, 3]]> : tensor<4x16xi8>
  %x_scale = arith.constant 0.5 : f32
  %x_zero_point = arith.constant 3 : i8
  %weight_scale = arith.constant dense<[0.25, 0.5, 0.125, 1.0]> : tensor<4xf32>
  %xdq_init = tensor.empty() : tensor<2x16xf32>
  %xdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]>
      ins(%xq, %x_scale, %x_zero_point : tensor<2x16xi8>, f32, i8)
      outs(%xdq_init : tensor<2x16xf32>) -> tensor<2x16xf32>
  %wdq_init = tensor.empty() : tensor<4x16xf32>
  %wdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(n, k) -> (n, k)>,
                        affine_map<(n, k) -> (n)>,
                        affine_map<(n, k) -> (n, k)>],
       quant_min = -8, quant_max = 7>
      ins(%weight, %weight_scale : tensor<4x16xi8>, tensor<4xf32>)
      outs(%wdq_init : tensor<4x16xf32>) -> tensor<4x16xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x4xf32>) -> tensor<2x4xf32>
  %result = linalg.matmul
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%xdq, %wdq : tensor<2x16xf32>, tensor<4x16xf32>)
      outs(%init : tensor<2x4xf32>) -> tensor<2x4xf32>
  check.expect_eq_const(%result, dense<[[-57.5, -20.75, -8.5625, 37.5], [-16.0, -19.5, -17.5625, -9.5]]> : tensor<2x4xf32>) : tensor<2x4xf32>
  return
}

// Unsigned 4-bit weights with a per-channel zero point: the contraction
// zero-extends the packed values.
func.func @unsigned_i4_weights_zero_point() {
  %xq = util.unfoldable_constant dense<[[18, -9, -12, 2, 8, -13, 1, 1, -12, 16, 4, -13, -14, -2, -4, -2], [14, 8, -14, 9, -8, 2, 19, -9, 10, 20, 12, -8, -9, 17, -1, -5]]> : tensor<2x16xi8>
  %weight = arith.constant dense<[[1, 7, 0, 10, 8, 11, 7, 12, 2, 6, 13, 6, 6, 6, 14, 11], [13, 15, 11, 2, 1, 15, 1, 0, 4, 9, 3, 3, 0, 14, 13, 15], [3, 10, 8, 9, 6, 4, 12, 5, 1, 4, 8, 15, 0, 1, 3, 11], [4, 2, 0, 14, 10, 0, 3, 9, 11, 13, 7, 8, 9, 8, 10, 15]]> : tensor<4x16xi8>
  %x_scale = arith.constant 0.5 : f32
  %weight_scale = arith.constant dense<[0.25, 0.5, 0.125, 1.0]> : tensor<4xf32>
  %weight_zero_point = arith.constant dense<[8, 1, 15, 0]> : tensor<4xi8>
  %xdq_init = tensor.empty() : tensor<2x16xf32>
  %xdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]>
      ins(%xq, %x_scale : tensor<2x16xi8>, f32)
      outs(%xdq_init : tensor<2x16xf32>) -> tensor<2x16xf32>
  %wdq_init = tensor.empty() : tensor<4x16xf32>
  %wdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(n, k) -> (n, k)>,
                        affine_map<(n, k) -> (n)>,
                        affine_map<(n, k) -> (n)>,
                        affine_map<(n, k) -> (n, k)>],
       quant_min = 0, quant_max = 15, input_unsigned, zp_unsigned>
      ins(%weight, %weight_scale, %weight_zero_point : tensor<4x16xi8>, tensor<4xf32>, tensor<4xi8>)
      outs(%wdq_init : tensor<4x16xf32>) -> tensor<4x16xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x4xf32>) -> tensor<2x4xf32>
  %result = linalg.matmul
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%xdq, %wdq : tensor<2x16xf32>, tensor<4x16xf32>)
      outs(%init : tensor<2x4xf32>) -> tensor<2x4xf32>
  check.expect_eq_const(%result, dense<[[4.375, -56.25, 13.5625, -19.0], [-10.75, 133.0, -37.25, 227.0]]> : tensor<2x4xf32>) : tensor<2x4xf32>
  return
}

// Signed 2-bit weights, packed four per byte.
func.func @signed_i2_weights() {
  %xq = util.unfoldable_constant dense<[[-18, 6, -14, 9, 0, 15, -10, -18, -6, 19, -2, 10, -19, -6, 2, -12], [-17, -14, -10, -9, -19, 3, 14, -18, -7, -2, 18, -5, -19, 6, 11, -11]]> : tensor<2x16xi8>
  %weight = arith.constant dense<[[-1, -2, -2, -1, 0, 1, 0, 1, 0, -2, -2, 0, -2, -1, 1, 0], [1, 0, 1, 1, -1, 0, -1, 0, 0, 1, 0, -1, -2, -2, 0, -2], [1, 1, 1, -2, -2, -1, -1, -1, -1, -2, 0, 1, -2, 1, 0, -1], [-2, -2, 1, -2, -1, 1, 1, 0, -2, 1, 1, 1, -1, -2, 1, -2]]> : tensor<4x16xi8>
  %x_scale = arith.constant 0.5 : f32
  %weight_scale = arith.constant dense<[0.25, 0.5, 0.125, 1.0]> : tensor<4xf32>
  %xdq_init = tensor.empty() : tensor<2x16xf32>
  %xdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]>
      ins(%xq, %x_scale : tensor<2x16xi8>, f32)
      outs(%xdq_init : tensor<2x16xf32>) -> tensor<2x16xf32>
  %wdq_init = tensor.empty() : tensor<4x16xf32>
  %wdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(n, k) -> (n, k)>,
                        affine_map<(n, k) -> (n)>,
                        affine_map<(n, k) -> (n, k)>],
       quant_min = -2, quant_max = 1>
      ins(%weight, %weight_scale : tensor<4x16xi8>, tensor<4xf32>)
      outs(%wdq_init : tensor<4x16xf32>) -> tensor<4x16xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x4xf32>) -> tensor<2x4xf32>
  %result = linalg.matmul
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%xdq, %wdq : tensor<2x16xf32>, tensor<4x16xf32>)
      outs(%init : tensor<2x4xf32>) -> tensor<2x4xf32>
  check.expect_eq_const(%result, dense<[[4.25, 17.5, -0.5625, 46.5], [8.75, 5.0, 4.8125, 85.5]]> : tensor<2x4xf32>) : tensor<2x4xf32>
  return
}

// Weight-only quantization. The unfolded scale keeps the dequantization out of
// constant evaluation, so it reads the packed weights at runtime.
func.func @signed_i4_weights_float_activations() {
  %x = util.unfoldable_constant dense<[[-2.0, -1.0, -1.5, -10.0, 9.5, 3.5, 2.0, -3.0, -2.0, -3.0, 3.5, 1.0, -7.5, 5.0, 9.5, 2.0], [-6.5, 0.5, -8.5, -4.5, -4.5, -8.5, -7.0, 3.0, -2.5, -7.5, -0.5, 1.5, -1.5, 4.0, 3.0, -1.0]]> : tensor<2x16xf32>
  %weight = arith.constant dense<[[-6, -2, -1, 7, 7, 1, -1, -2, -1, -3, 4, 7, 6, -2, -5, 4], [-4, -4, -6, -6, -5, -7, 0, 0, 0, -3, -3, 5, -7, 2, 0, 4], [3, -1, 4, -7, 1, 4, -6, 1, -4, -6, 5, 3, -3, -8, 7, 0], [-5, 5, -1, -5, 5, 1, 0, -7, -2, 1, 0, 7, -5, -2, 5, -2]]> : tensor<4x16xi8>
  %weight_scale = util.unfoldable_constant dense<[0.25, 0.5, 0.125, 1.0]> : tensor<4xf32>
  %wdq_init = tensor.empty() : tensor<4x16xf32>
  %wdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(n, k) -> (n, k)>,
                        affine_map<(n, k) -> (n)>,
                        affine_map<(n, k) -> (n, k)>],
       quant_min = -8, quant_max = 7>
      ins(%weight, %weight_scale : tensor<4x16xi8>, tensor<4xf32>)
      outs(%wdq_init : tensor<4x16xf32>) -> tensor<4x16xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x4xf32>) -> tensor<2x4xf32>
  %result = linalg.matmul
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%x, %wdq : tensor<2x16xf32>, tensor<4x16xf32>)
      outs(%init : tensor<2x4xf32>) -> tensor<2x4xf32>
  check.expect_eq_const(%result, dense<[[-10.75, 41.5, 20.375, 207.5], [-6.625, 115.0, 4.3125, 38.5]]> : tensor<2x4xf32>) : tensor<2x4xf32>
  return
}
