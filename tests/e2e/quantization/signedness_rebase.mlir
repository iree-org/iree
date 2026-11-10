// Contractions with unsigned operands that CPU data tiling tiles only after
// they are rebased to signed storage. The inputs are unfoldable constants, so
// the compiled contractions run.

// A u8 activation quantized at run time times a constant s8 weight.
func.func @u8_activation() {
  %x = util.unfoldable_constant dense<[[1.0, 2.0, -1.0, 0.5], [-2.0, 0.0, 3.0, 1.5]]> : tensor<2x4xf32>
  %sx = util.unfoldable_constant 0.5 : f32
  %zx = util.unfoldable_constant 100 : i8
  %w = arith.constant dense<[[1, -2, 3, 4], [5, 6, -7, 8], [-1, 0, 2, -3]]> : tensor<3x4xi8>
  %sw = arith.constant 1.0 : f32
  %qi = tensor.empty() : tensor<2x4xi8>
  %xq = iree_linalg_ext.quantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>],
       quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>
      ins(%x, %sx, %zx : tensor<2x4xf32>, f32, i8) outs(%qi : tensor<2x4xi8>) -> tensor<2x4xi8>
  %xi = tensor.empty() : tensor<2x4xf32>
  %xd = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>],
       input_unsigned, zp_unsigned>
      ins(%xq, %sx, %zx : tensor<2x4xi8>, f32, i8) outs(%xi : tensor<2x4xf32>) -> tensor<2x4xf32>
  %wi = tensor.empty() : tensor<3x4xf32>
  %wd = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%w, %sw : tensor<3x4xi8>, f32) outs(%wi : tensor<3x4xf32>) -> tensor<3x4xf32>
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<2x3xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<2x3xf32>) -> tensor<2x3xf32>
  %result = linalg.matmul
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%xd, %wd : tensor<2x4xf32>, tensor<3x4xf32>) outs(%f : tensor<2x3xf32>) -> tensor<2x3xf32>
  check.expect_eq_const(%result, dense<[[-4.0, 28.0, -4.5], [13.0, -19.0, 3.5]]> : tensor<2x3xf32>) : tensor<2x3xf32>
  return
}

// A constant u4 weight in u8 storage with a zero point per output channel, times
// a symmetric s8 activation.
func.func @u4_weight() {
  %x = util.unfoldable_constant dense<[[1.0, -2.0, 3.0, 4.0], [0.0, 1.0, -1.0, 2.0]]> : tensor<2x4xf32>
  %sx = util.unfoldable_constant 1.0 : f32
  %w = arith.constant dense<[[0, 15, 8, 9], [3, 12, 7, 1]]> : tensor<2x4xi8>
  %sw = arith.constant dense<[0.5, 2.0]> : tensor<2xf32>
  %zw = arith.constant dense<[8, 6]> : tensor<2xi8>
  %qi = tensor.empty() : tensor<2x4xi8>
  %xq = iree_linalg_ext.quantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       quant_min = -128, quant_max = 127>
      ins(%x, %sx : tensor<2x4xf32>, f32) outs(%qi : tensor<2x4xi8>) -> tensor<2x4xi8>
  %xi = tensor.empty() : tensor<2x4xf32>
  %xd = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%xq, %sx : tensor<2x4xi8>, f32) outs(%xi : tensor<2x4xf32>) -> tensor<2x4xf32>
  %wi = tensor.empty() : tensor<2x4xf32>
  %wd = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(n, k) -> (n, k)>, affine_map<(n, k) -> (n)>,
                        affine_map<(n, k) -> (n)>, affine_map<(n, k) -> (n, k)>],
       quant_min = 0, quant_max = 15, input_unsigned, zp_unsigned>
      ins(%w, %sw, %zw : tensor<2x4xi8>, tensor<2xf32>, tensor<2xi8>) outs(%wi : tensor<2x4xf32>) -> tensor<2x4xf32>
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<2x2xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<2x2xf32>) -> tensor<2x2xf32>
  %result = linalg.matmul
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%xd, %wd : tensor<2x4xf32>, tensor<2x4xf32>) outs(%f : tensor<2x2xf32>) -> tensor<2x2xf32>
  check.expect_eq_const(%result, dense<[[-9.0, -64.0], [4.5, -10.0]]> : tensor<2x2xf32>) : tensor<2x2xf32>
  return
}
