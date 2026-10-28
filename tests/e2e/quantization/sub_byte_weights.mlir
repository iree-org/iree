// Contractions over constant weights whose quantization range fits 2 or 4
// bits. With --iree-global-opt-enable-sub-byte-quantized-storage the weights
// are packed into i4 storage; without it they stay i8. Both must give the same
// results. Scales are powers of two, so the expected values are exact.

// Signed 4-bit weights with quantized activations: an i8 x i4 integer
// contraction.
func.func @signed_i4_weights() {
  %x = util.unfoldable_constant dense<[[9.5, -9.0, -1.5, -8.0, 0.0, 0.5, -3.5, 8.5, 1.0, -8.0, 9.0, -10.0, -8.0, 2.5, -3.0, -4.5], [1.0, -2.5, 10.0, 5.0, -6.0, 1.5, 5.5, 10.0, -6.5, -0.5, -2.0, -7.5, 4.0, -10.0, -2.0, 8.5]]> : tensor<2x16xf32>
  %weight = arith.constant dense<[[-3, 0, 3, 4, 0, -2, -1, -2, 1, -6, 5, 7, -8, 3, -3, -3], [6, -8, -7, -1, -3, 6, -7, 3, -1, -2, -6, -6, -8, -8, 3, -2], [-6, -6, -3, 1, -4, 7, 5, 4, 2, -8, 3, -6, -5, -6, 7, -5], [-4, -3, -7, 7, 0, -8, -3, 0, -4, -3, -7, 6, 4, 1, 6, 2]]> : tensor<4x16xi8>
  %weight_scale = arith.constant dense<[0.25, 0.5, 0.125, 1.0]> : tensor<4xf32>
  %x_scale = arith.constant 0.5 : f32
  %xq_init = tensor.empty() : tensor<2x16xi8>
  %xq = iree_linalg_ext.quantize_affine
      <indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>],
       quant_min = -128, quant_max = 127>
      ins(%x, %x_scale : tensor<2x16xf32>, f32)
      outs(%xq_init : tensor<2x16xi8>) -> tensor<2x16xi8>
  %xdq_init = tensor.empty() : tensor<2x16xf32>
  %xdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>],
       quant_min = -128, quant_max = 127>
      ins(%xq, %x_scale : tensor<2x16xi8>, f32)
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
  check.expect_eq_const(%result, dense<[[9.625, 132.75, 24.125, -209.5], [-32.25, 29.5, 12.4375, -52.5]]> : tensor<2x4xf32>) : tensor<2x4xf32>
  return
}

// Unsigned 4-bit weights with a per-channel zero point. Packing stores the
// weights and the zero point minus 8.
func.func @unsigned_i4_weights_zero_point() {
  %x = util.unfoldable_constant dense<[[9.0, -4.5, -6.0, 1.0, 4.0, -6.5, 0.5, 0.5, -6.0, 8.0, 2.0, -6.5, -7.0, -1.0, -2.0, -1.0], [7.0, 4.0, -7.0, 4.5, -4.0, 1.0, 9.5, -4.5, 5.0, 10.0, 6.0, -4.0, -4.5, 8.5, -0.5, -2.5]]> : tensor<2x16xf32>
  %weight = arith.constant dense<[[1, 7, 0, 10, 8, 11, 7, 12, 2, 6, 13, 6, 6, 6, 14, 11], [13, 15, 11, 2, 1, 15, 1, 0, 4, 9, 3, 3, 0, 14, 13, 15], [3, 10, 8, 9, 6, 4, 12, 5, 1, 4, 8, 15, 0, 1, 3, 11], [4, 2, 0, 14, 10, 0, 3, 9, 11, 13, 7, 8, 9, 8, 10, 15]]> : tensor<4x16xi8>
  %weight_scale = arith.constant dense<[0.25, 0.5, 0.125, 1.0]> : tensor<4xf32>
  %zero_point = arith.constant dense<[8, 1, 15, 0]> : tensor<4xi8>
  %x_scale = arith.constant 0.5 : f32
  %xq_init = tensor.empty() : tensor<2x16xi8>
  %xq = iree_linalg_ext.quantize_affine
      <indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>],
       quant_min = -128, quant_max = 127>
      ins(%x, %x_scale : tensor<2x16xf32>, f32)
      outs(%xq_init : tensor<2x16xi8>) -> tensor<2x16xi8>
  %xdq_init = tensor.empty() : tensor<2x16xf32>
  %xdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>],
       quant_min = -128, quant_max = 127>
      ins(%xq, %x_scale : tensor<2x16xi8>, f32)
      outs(%xdq_init : tensor<2x16xf32>) -> tensor<2x16xf32>
  %wdq_init = tensor.empty() : tensor<4x16xf32>
  %wdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(n, k) -> (n, k)>,
                        affine_map<(n, k) -> (n)>,
                        affine_map<(n, k) -> (n)>,
                        affine_map<(n, k) -> (n, k)>],
       quant_min = 0, quant_max = 15, input_unsigned, zp_unsigned>
      ins(%weight, %weight_scale, %zero_point : tensor<4x16xi8>, tensor<4xf32>, tensor<4xi8>)
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

// Unsigned 4-bit weights with the MatMulNBits default zero point 8. Packing
// stores them minus 8, which leaves a zero point of 0 and drops it.
func.func @unsigned_i4_weights_default_zero_point() {
  %x = util.unfoldable_constant dense<[[-5.5, 4.5, 5.5, 8.5, -4.5, 8.5, -8.0, 0.0, -7.5, -6.0, -1.0, -1.5, -3.0, -8.0, -3.0, -6.0], [2.5, 6.5, -0.5, -8.5, 3.0, -10.0, -3.0, -2.0, 5.0, -9.0, 2.5, -1.5, -10.0, -5.5, 7.5, -0.5]]> : tensor<2x16xf32>
  %weight = arith.constant dense<[[1, 7, 1, 5, 5, 2, 2, 13, 15, 11, 14, 1, 8, 13, 8, 10], [14, 15, 8, 4, 14, 6, 0, 4, 14, 6, 12, 13, 1, 5, 10, 8], [8, 4, 11, 10, 7, 3, 10, 10, 0, 7, 1, 15, 9, 13, 2, 9], [2, 3, 9, 1, 13, 13, 11, 5, 11, 1, 0, 11, 0, 8, 15, 10]]> : tensor<4x16xi8>
  %weight_scale = arith.constant dense<[0.25, 0.5, 0.125, 1.0]> : tensor<4xf32>
  %zero_point = arith.constant dense<8> : tensor<4xi8>
  %x_scale = arith.constant 0.5 : f32
  %xq_init = tensor.empty() : tensor<2x16xi8>
  %xq = iree_linalg_ext.quantize_affine
      <indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>],
       quant_min = -128, quant_max = 127>
      ins(%x, %x_scale : tensor<2x16xf32>, f32)
      outs(%xq_init : tensor<2x16xi8>) -> tensor<2x16xi8>
  %xdq_init = tensor.empty() : tensor<2x16xf32>
  %xdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>],
       quant_min = -128, quant_max = 127>
      ins(%xq, %x_scale : tensor<2x16xi8>, f32)
      outs(%xdq_init : tensor<2x16xf32>) -> tensor<2x16xf32>
  %wdq_init = tensor.empty() : tensor<4x16xf32>
  %wdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(n, k) -> (n, k)>,
                        affine_map<(n, k) -> (n)>,
                        affine_map<(n, k) -> (n)>,
                        affine_map<(n, k) -> (n, k)>],
       quant_min = 0, quant_max = 15, input_unsigned, zp_unsigned>
      ins(%weight, %weight_scale, %zero_point : tensor<4x16xi8>, tensor<4xf32>, tensor<4xi8>)
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
  check.expect_eq_const(%result, dense<[[-34.375, -10.5, -0.875, -33.5], [17.25, 158.25, -18.6875, 158.5]]> : tensor<2x4xf32>) : tensor<2x4xf32>
  return
}

// Activations with a zero point: the correction term sums the packed
// weight values.
func.func @signed_i4_weights_asymmetric_activations() {
  %x = util.unfoldable_constant dense<[[-4.5, 6.5, -7.5, -2.5, -9.0, -0.5, 4.0, 3.5, -11.0, -9.0, -0.5, 3.0, -8.0, -4.5, -11.5, 4.0], [-2.0, -10.5, -6.5, -6.5, -4.0, -10.5, -4.0, 7.5, 7.5, 1.5, 6.0, -0.5, 8.5, -4.0, -1.0, 7.0]]> : tensor<2x16xf32>
  %weight = arith.constant dense<[[6, -2, 4, 5, 6, 5, -1, 5, 7, 2, 4, -3, -5, -7, 4, -6], [-5, -6, 4, 5, 7, -5, -1, 4, -3, 0, -2, 7, -6, 0, 2, -3], [4, 7, 7, -6, 3, -6, -8, -7, 3, -2, 2, 0, -5, 7, -5, -7], [-6, 1, -1, -6, 7, 1, -3, -7, 5, -7, 6, 1, -8, -2, 1, 3]]> : tensor<4x16xi8>
  %weight_scale = arith.constant dense<[0.25, 0.5, 0.125, 1.0]> : tensor<4xf32>
  %x_scale = arith.constant 0.5 : f32
  %x_zero_point = arith.constant 3 : i8
  %xq_init = tensor.empty() : tensor<2x16xi8>
  %xq = iree_linalg_ext.quantize_affine
      <indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>],
       quant_min = -128, quant_max = 127>
      ins(%x, %x_scale, %x_zero_point : tensor<2x16xf32>, f32, i8)
      outs(%xq_init : tensor<2x16xi8>) -> tensor<2x16xi8>
  %xdq_init = tensor.empty() : tensor<2x16xf32>
  %xdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>],
       quant_min = -128, quant_max = 127>
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

// Signed 2-bit weights. The pipeline packs them into i4, since i2 storage is
// only used with the pass option min-bit-width=2.
func.func @signed_i2_weights() {
  %x = util.unfoldable_constant dense<[[-9.0, 3.0, -7.0, 4.5, 0.0, 7.5, -5.0, -9.0, -3.0, 9.5, -1.0, 5.0, -9.5, -3.0, 1.0, -6.0], [-8.5, -7.0, -5.0, -4.5, -9.5, 1.5, 7.0, -9.0, -3.5, -1.0, 9.0, -2.5, -9.5, 3.0, 5.5, -5.5]]> : tensor<2x16xf32>
  %weight = arith.constant dense<[[-1, -2, -2, -1, 0, 1, 0, 1, 0, -2, -2, 0, -2, -1, 1, 0], [1, 0, 1, 1, -1, 0, -1, 0, 0, 1, 0, -1, -2, -2, 0, -2], [1, 1, 1, -2, -2, -1, -1, -1, -1, -2, 0, 1, -2, 1, 0, -1], [-2, -2, 1, -2, -1, 1, 1, 0, -2, 1, 1, 1, -1, -2, 1, -2]]> : tensor<4x16xi8>
  %weight_scale = arith.constant dense<[0.25, 0.5, 0.125, 1.0]> : tensor<4xf32>
  %x_scale = arith.constant 0.5 : f32
  %xq_init = tensor.empty() : tensor<2x16xi8>
  %xq = iree_linalg_ext.quantize_affine
      <indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>],
       quant_min = -128, quant_max = 127>
      ins(%x, %x_scale : tensor<2x16xf32>, f32)
      outs(%xq_init : tensor<2x16xi8>) -> tensor<2x16xi8>
  %xdq_init = tensor.empty() : tensor<2x16xf32>
  %xdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>],
       quant_min = -128, quant_max = 127>
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

// Weight-only quantization. A weight this small is dequantized by constant
// evaluation, whose JIT-compiled program reads the packed values.
func.func @signed_i4_weights_float_activations() {
  %x = util.unfoldable_constant dense<[[-2.0, -1.0, -1.5, -10.0, 9.5, 3.5, 2.0, -3.0, -2.0, -3.0, 3.5, 1.0, -7.5, 5.0, 9.5, 2.0], [-6.5, 0.5, -8.5, -4.5, -4.5, -8.5, -7.0, 3.0, -2.5, -7.5, -0.5, 1.5, -1.5, 4.0, 3.0, -1.0]]> : tensor<2x16xf32>
  %weight = arith.constant dense<[[-6, -2, -1, 7, 7, 1, -1, -2, -1, -3, 4, 7, 6, -2, -5, 4], [-4, -4, -6, -6, -5, -7, 0, 0, 0, -3, -3, 5, -7, 2, 0, 4], [3, -1, 4, -7, 1, 4, -6, 1, -4, -6, 5, 3, -3, -8, 7, 0], [-5, 5, -1, -5, 5, 1, 0, -7, -2, 1, 0, 7, -5, -2, 5, -2]]> : tensor<4x16xi8>
  %weight_scale = arith.constant dense<[0.25, 0.5, 0.125, 1.0]> : tensor<4xf32>
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

// A batch matmul with a batch of one. Dropping the unit batch dimension
// reshapes the packed weights.
func.func @signed_i4_weights_unit_batch() {
  %x = util.unfoldable_constant dense<[[[7.0, 0.5, 8.0, -0.5, -7.0, -2.5, -2.0, 1.5, -2.5, 4.0, -8.0, 10.0, -6.5, 6.5, -4.5, 0.5], [-4.0, 9.5, -1.0, -8.0, 1.5, -6.0, -8.0, 3.5, 0.5, -6.0, -3.5, -8.5, -1.0, 2.5, -1.0, -3.5]]]> : tensor<1x2x16xf32>
  %weight = arith.constant dense<[[[-1, 3, 4, -2, 5, -1, 3, -8, 7, -5, 3, -2, 7, -8, 5, 3], [-7, 6, -8, -3, 0, -1, -2, 6, 0, 3, 2, 4, -7, -5, 3, 3], [-8, 6, -5, 5, 4, -7, -6, -6, 4, -1, 0, -3, -8, 3, 4, 3], [-6, 6, -8, 1, -5, 1, -1, 0, 7, 0, -2, -7, 3, 5, 0, 0]]]> : tensor<1x4x16xi8>
  %weight_scale = arith.constant dense<[[0.25, 0.5, 0.125, 1.0]]> : tensor<1x4xf32>
  %x_scale = arith.constant 0.5 : f32
  %xq_init = tensor.empty() : tensor<1x2x16xi8>
  %xq = iree_linalg_ext.quantize_affine
      <indexing_maps = [affine_map<(b, m, k) -> (b, m, k)>,
                        affine_map<(b, m, k) -> ()>,
                        affine_map<(b, m, k) -> (b, m, k)>],
       quant_min = -128, quant_max = 127>
      ins(%x, %x_scale : tensor<1x2x16xf32>, f32)
      outs(%xq_init : tensor<1x2x16xi8>) -> tensor<1x2x16xi8>
  %xdq_init = tensor.empty() : tensor<1x2x16xf32>
  %xdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(b, m, k) -> (b, m, k)>,
                        affine_map<(b, m, k) -> ()>,
                        affine_map<(b, m, k) -> (b, m, k)>],
       quant_min = -128, quant_max = 127>
      ins(%xq, %x_scale : tensor<1x2x16xi8>, f32)
      outs(%xdq_init : tensor<1x2x16xf32>) -> tensor<1x2x16xf32>
  %wdq_init = tensor.empty() : tensor<1x4x16xf32>
  %wdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(b, n, k) -> (b, n, k)>,
                        affine_map<(b, n, k) -> (b, n)>,
                        affine_map<(b, n, k) -> (b, n, k)>],
       quant_min = -8, quant_max = 7>
      ins(%weight, %weight_scale : tensor<1x4x16xi8>, tensor<1x4xf32>)
      outs(%wdq_init : tensor<1x4x16xf32>) -> tensor<1x4x16xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x2x4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
  %result = linalg.batch_matmul
      indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                       affine_map<(b, m, n, k) -> (b, n, k)>,
                       affine_map<(b, m, n, k) -> (b, m, n)>]
      ins(%xdq, %wdq : tensor<1x2x16xf32>, tensor<1x4x16xf32>)
      outs(%init : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
  check.expect_eq_const(%result, dense<[[[-55.75, -28.0, -11.5, -127.5], [0.875, 41.0, 20.4375, 155.0]]]> : tensor<1x2x4xf32>) : tensor<1x2x4xf32>
  return
}
