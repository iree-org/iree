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

// NCHW layout with two input channels: all three filter dimensions reduce.
// Distinct signed data and per-filter parameters expose channel/layout swaps.
func.func @convolution_nchw_asymmetric() {
  %aq = util.unfoldable_constant dense<[[[[-8, -5, -2, 1], [-3, 0, 3, 6], [2, 5, 8, -6]], [[3, 6, -8, -5], [8, -6, -3, 0], [-4, -1, 2, 5]]]]> : tensor<1x2x3x4xi8>
  %bq = util.unfoldable_constant dense<[[[[-5, -4], [-2, -1]], [[0, 1], [3, 4]]], [[[2, 3], [5, -5]], [[-4, -3], [-1, 0]]]]> : tensor<2x2x2x2xi8>
  %sb = util.unfoldable_constant dense<[1.0, 0.25]> : tensor<2xf32>
  %zb = util.unfoldable_constant dense<[-1, 2]> : tensor<2xi8>
  %sa = util.unfoldable_constant 0.5 : f32
  %za = util.unfoldable_constant -2 : i8
  %ai = tensor.empty() : tensor<1x2x3x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(n, c, h, w) -> (n, c, h, w)>,
                        affine_map<(n, c, h, w) -> ()>,
                        affine_map<(n, c, h, w) -> ()>,
                        affine_map<(n, c, h, w) -> (n, c, h, w)>]}
      ins(%aq, %sa, %za : tensor<1x2x3x4xi8>, f32, i8)
      outs(%ai : tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  %bi = tensor.empty() : tensor<2x2x2x2xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(f, c, h, w) -> (f, c, h, w)>,
                        affine_map<(f, c, h, w) -> (f)>,
                        affine_map<(f, c, h, w) -> (f)>,
                        affine_map<(f, c, h, w) -> (f, c, h, w)>]}
      ins(%bq, %sb, %zb : tensor<2x2x2x2xi8>, tensor<2xf32>, tensor<2xi8>)
      outs(%bi : tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x2x2x3xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%a, %b : tensor<1x2x3x4xf32>, tensor<2x2x2x2xf32>)
      outs(%init : tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
  check.expect_eq_const(%result, dense<[[[[37.5, -7.5, -10.0], [-3.5, -6.0, 0.0]], [[-14.0, -4.125, 1.5], [-8.875, -3.25, 4.5]]]]> : tensor<1x2x2x3xf32>) : tensor<1x2x2x3xf32>
  return
}

// Channels remain parallel in depthwise convolution. Both operands have
// channel-specific scales and zero points; activation bytes exercise extui.
// Different stride/dilation axes distinguish the sampled spatial windows.
func.func @depthwise_asymmetric_strided_dilated() {
  %aq = util.unfoldable_constant dense<[[[[128, 133], [131, 136], [134, 139], [137, 142], [140, 145]], [[135, 140], [138, 143], [141, 146], [144, 130], [128, 133]], [[142, 128], [145, 131], [129, 134], [132, 137], [135, 140]], [[130, 135], [133, 138], [136, 141], [139, 144], [142, 128]]]]> : tensor<1x4x5x2xi8>
  %bq = util.unfoldable_constant dense<[[[-3, 4], [2, -1]], [[5, 0], [-2, 6]]]> : tensor<2x2x2xi8>
  %sa = util.unfoldable_constant dense<[0.5, 2.0]> : tensor<2xf32>
  %za = util.unfoldable_constant dense<[132, 136]> : tensor<2xi8>
  %sb = util.unfoldable_constant dense<[2.0, 0.25]> : tensor<2xf32>
  %zb = util.unfoldable_constant dense<[-1, 2]> : tensor<2xi8>
  %ai = tensor.empty() : tensor<1x4x5x2xf32>
  %a = iree_linalg_ext.dequantize_affine
      {input_unsigned, zp_unsigned, indexing_maps = [affine_map<(n, h, w, c) -> (n, h, w, c)>,
                        affine_map<(n, h, w, c) -> (c)>,
                        affine_map<(n, h, w, c) -> (c)>,
                        affine_map<(n, h, w, c) -> (n, h, w, c)>]}
      ins(%aq, %sa, %za : tensor<1x4x5x2xi8>, tensor<2xf32>, tensor<2xi8>)
      outs(%ai : tensor<1x4x5x2xf32>) -> tensor<1x4x5x2xf32>
  %bi = tensor.empty() : tensor<2x2x2xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(h, w, c) -> (h, w, c)>,
                        affine_map<(h, w, c) -> (c)>,
                        affine_map<(h, w, c) -> (c)>,
                        affine_map<(h, w, c) -> (h, w, c)>]}
      ins(%bq, %sb, %zb : tensor<2x2x2xi8>, tensor<2xf32>, tensor<2xi8>)
      outs(%bi : tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x2x2x2xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %result = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<[2, 1]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>}
      ins(%a, %b : tensor<1x4x5x2xf32>, tensor<2x2x2xf32>)
      outs(%init : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  check.expect_eq_const(%result, dense<[[[[52.0, -5.0], [-7.0, -2.0]], [[-1.0, -1.5], [35.0, 30.0]]]]> : tensor<1x2x2x2xf32>) : tensor<1x2x2x2xf32>
  return
}

// A custom strided window contraction with interleaved reduction/parallel
// loops, a broadcast RHS batch, and output axes ordered (n, batch, m).
// C[n,t,m] = sum_{r,c} DQ(A[t,2*m+r,c]) * DQ(B[n,c,r]).
func.func @generic_strided_permuted_contraction() {
  %aq = util.unfoldable_constant dense<[[[128, 131], [132, 135], [136, 139], [140, 143], [144, 147], [148, 128], [129, 132]], [[137, 140], [141, 144], [145, 148], [149, 129], [130, 133], [134, 137], [138, 141]]]> : tensor<2x7x2xi8>
  %bq = util.unfoldable_constant dense<[[[-6, -4, -2], [-3, -1, 1]], [[-1, 1, 3], [2, 4, 6]], [[4, 6, -5], [-6, -4, -2]]]> : tensor<3x2x3xi8>
  %sa = util.unfoldable_constant dense<[0.5, 1.0]> : tensor<2xf32>
  %za = util.unfoldable_constant dense<[130, 134]> : tensor<2xi8>
  %sb = util.unfoldable_constant dense<[1.0, 0.5, 2.0]> : tensor<3xf32>
  %zb = util.unfoldable_constant dense<[-2, 1, 3]> : tensor<3xi8>
  %ai = tensor.empty() : tensor<2x7x2xf32>
  %a = iree_linalg_ext.dequantize_affine
      {input_unsigned, zp_unsigned, indexing_maps = [affine_map<(t, x, c) -> (t, x, c)>,
                        affine_map<(t, x, c) -> (t)>,
                        affine_map<(t, x, c) -> (t)>,
                        affine_map<(t, x, c) -> (t, x, c)>]}
      ins(%aq, %sa, %za : tensor<2x7x2xi8>, tensor<2xf32>, tensor<2xi8>)
      outs(%ai : tensor<2x7x2xf32>) -> tensor<2x7x2xf32>
  %bi = tensor.empty() : tensor<3x2x3xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(n, c, r) -> (n, c, r)>,
                        affine_map<(n, c, r) -> (n)>,
                        affine_map<(n, c, r) -> (n)>,
                        affine_map<(n, c, r) -> (n, c, r)>]}
      ins(%bq, %sb, %zb : tensor<3x2x3xi8>, tensor<3xf32>, tensor<3xi8>)
      outs(%bi : tensor<3x2x3xf32>) -> tensor<3x2x3xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<3x2x3xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<3x2x3xf32>) -> tensor<3x2x3xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(c, n, t, r, m) -> (t, m * 2 + r, c)>,
                       affine_map<(c, n, t, r, m) -> (n, c, r)>,
                       affine_map<(c, n, t, r, m) -> (n, t, m)>],
      iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel"]}
      ins(%a, %b : tensor<2x7x2xf32>, tensor<3x2x3xf32>)
      outs(%init : tensor<3x2x3xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
  } -> tensor<3x2x3xf32>
  check.expect_eq_const(%result, dense<[[[17.5, 5.5, -52.5], [20.0, -96.0, 41.0]], [[19.25, 37.25, -2.25], [61.0, -18.0, 29.5]], [[-133.0, -333.0, -73.0], [-516.0, 4.0, -166.0]]]> : tensor<3x2x3xf32>) : tensor<3x2x3xf32>
  return
}

// Both asymmetric operands vary by block and output axis, with different
// storage permutations folded into dequantization. Only k is reduced in i32;
// g is retained in partials and reduced after applying both block scales.
// C[n,m] = sum_{g,k} DQ(A[k,m,g]) * DQ(B[n,k,g]).
func.func @blockwise_asymmetric_permuted_storage() {
  %aq = util.unfoldable_constant dense<[[[128, 131], [133, 136]], [[135, 138], [140, 143]], [[142, 145], [147, 129]]]> : tensor<3x2x2xi8>
  %bq = util.unfoldable_constant dense<[[[-8, -3], [-5, 0], [-2, 3]], [[-1, 4], [2, 7], [5, -7]], [[6, -6], [-8, -3], [-5, 0]]]> : tensor<3x3x2xi8>
  %sa = util.unfoldable_constant dense<[[0.5, 2.0], [1.0, 0.25]]> : tensor<2x2xf32>
  %za = util.unfoldable_constant dense<[[130, 135], [134, 129]]> : tensor<2x2xi8>
  %sb = util.unfoldable_constant dense<[[1.0, 0.5], [2.0, 0.25], [0.5, 2.0]]> : tensor<3x2xf32>
  %zb = util.unfoldable_constant dense<[[-3, 1], [2, -4], [-1, 3]]> : tensor<3x2xi8>
  %ai = tensor.empty() : tensor<2x2x3xf32>
  %a = iree_linalg_ext.dequantize_affine
      {input_unsigned, zp_unsigned, indexing_maps = [affine_map<(m, g, k) -> (k, m, g)>,
                        affine_map<(m, g, k) -> (g, m)>,
                        affine_map<(m, g, k) -> (g, m)>,
                        affine_map<(m, g, k) -> (m, g, k)>]}
      ins(%aq, %sa, %za : tensor<3x2x2xi8>, tensor<2x2xf32>, tensor<2x2xi8>)
      outs(%ai : tensor<2x2x3xf32>) -> tensor<2x2x3xf32>
  %bi = tensor.empty() : tensor<2x3x3xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(g, k, n) -> (n, k, g)>,
                        affine_map<(g, k, n) -> (n, g)>,
                        affine_map<(g, k, n) -> (n, g)>,
                        affine_map<(g, k, n) -> (g, k, n)>]}
      ins(%bq, %sb, %zb : tensor<3x3x2xi8>, tensor<3x2xf32>, tensor<3x2xi8>)
      outs(%bi : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<3x2xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<3x2xf32>) -> tensor<3x2xf32>
  %result = linalg.contract
      indexing_maps = [affine_map<(m, n, g, k) -> (m, g, k)>,
                       affine_map<(m, n, g, k) -> (g, k, n)>,
                       affine_map<(m, n, g, k) -> (n, m)>]
      ins(%a, %b : tensor<2x2x3xf32>, tensor<2x3x3xf32>)
      outs(%init : tensor<3x2xf32>) -> tensor<3x2xf32>
  check.expect_eq_const(%result, dense<[[21.0, 18.75], [38.75, 181.125], [-84.25, -170.5]]> : tensor<3x2xf32>) : tensor<3x2xf32>
  return
}

// Anticipated QDQ Linalg translation of a PT2E/XNNPACK bias-free Linear(16,16)
// with an 8x16 input: Q -> DQ -> matmul(input, DQ(weight).T) -> Q -> DQ.
// Input, quantized weights, scales, zero points, and expected output are from
// a seed-0 PyTorch run. The RHS indexing map incorporates aten.transpose.int.
func.func @pt2e_linear_qdq() {
  %input_scale = arith.constant 0.024613387882709503 : f32
  %weight_scale = arith.constant 0.0019636377692222595 : f32
  %output_scale = arith.constant 0.010716261342167854 : f32
  %input_zp = arith.constant -33 : i64
  %weight_zp = arith.constant 0 : i64
  %output_zp = arith.constant -10 : i64
  %zero = arith.constant 0.0 : f32

  // Prevent constant folding so the QDQ graph executes at runtime.
  %input = util.unfoldable_constant dense<[
      [-0.03937254101037979, -0.8014721870422363, -0.49554431438446045, -0.36151406168937683, 0.5851132273674011, -1.1560065746307373, -0.14336487650871277, -0.19474059343338013, -0.08556340634822845, 1.3945199251174927, 0.5969000458717346, -0.4828483462333679, -0.3660986125469208, -1.3270524740219116, 1.695279598236084, 2.0654995441436768],
      [-0.2339576780796051, 0.7073183059692383, 0.5800480842590332, 0.2683020830154419, -2.05893874168396, 0.5340206623077393, -0.5353949069976807, -0.863664984703064, -0.023494405671954155, 1.171669840812683, 0.3986871838569641, -0.19871552288532257, -1.1559407711029053, -0.3166695237159729, 0.9402980804443359, -1.1469545364379883],
      [0.5588033199310303, 0.7917584776878357, -0.18467576801776886, -0.7317724823951721, -0.0806519165635109, -0.9800607562065125, 0.060491468757390976, -0.4889546036720276, -0.8137312531471252, 0.8199948072433472, -0.633173406124115, 1.2947547435760498, 1.4628292322158813, -0.6204336285591125, 0.988389253616333, -0.4321781396865845],
      [-0.6232188940048218, -0.21624533832073212, -0.48867544531822205, 0.7869559526443481, 0.10759072750806808, -1.0714776515960693, -0.11664776504039764, -1.0169707536697388, -1.1979783773422241, 0.47843775153160095, -1.2295243740081787, -1.3700363636016846, 1.5435417890548706, -0.0332069993019104, -0.4186263680458069, -0.25559672713279724],
      [-0.12923042476177216, -0.05459475517272949, 0.4083467423915863, 1.1263659000396729, 1.9350574016571045, 1.0076850652694702, 1.0046416521072388, -0.43351978063583374, -1.2425976991653442, 1.2845500707626343, 0.24377228319644928, 0.5303686261177063, -0.014530729502439499, -2.235718250274658, 1.4660301208496094, -1.2190581560134888],
      [0.64423006772995, 3.9300038814544678, -0.1244242936372757, 0.29534170031547546, 0.38265419006347656, -0.549721360206604, -0.9940357208251953, 1.345936894416809, 1.9456682205200195, -1.2903639078140259, -2.3494760990142822, -2.068861961364746, 0.9094210863113403, -0.6946200728416443, 1.9594571590423584, -1.1038278341293335],
      [0.5411418080329895, 1.5389583110809326, 1.0860490798950195, 1.246405005455017, 0.11507505923509598, 1.619307518005371, 0.4636935293674469, 1.3007354736328125, 0.873230516910553, 0.0651267021894455, 0.7732411026954651, -0.970138430595398, -0.8876768946647644, -0.3183162808418274, -0.3344036936759949, 0.4542836546897888],
      [0.49895304441452026, 0.8779974579811096, 0.38944435119628906, 1.462517499923706, 0.47950610518455505, -0.5333999991416931, -0.03465134650468826, 0.6572969555854797, -0.3112243115901947, -0.5620035529136658, -0.4834926426410675, -1.2721126079559326, -0.17401821911334991, 0.5541167259216309, -0.18165524303913116, -0.23447339236736298]
  ]> : tensor<8x16xf32>
  // Signed bytes decoded from torch_tensor_16_16_torch.int8, excluding its
  // four-byte resource alignment header. Rows retain Torch weight layout [N,K].
  %weight = util.unfoldable_constant dense<[
      [-1, 68, -105, -94, -49, 34, -3, 101, -11, 34, -38, -25, -122, -84, -52, 5],
      [50, 76, -86, -55, 46, 106, -26, 95, -21, 13, 115, -118, -80, -32, -50, 110],
      [-83, -59, -89, -119, -74, 109, 57, 62, 7, -65, 22, -119, -92, -66, 80, 75],
      [-56, -5, 81, 127, 51, 17, 85, -75, 24, -99, -88, -66, 58, 51, -75, 38],
      [70, -16, 5, 29, 79, 122, -98, -47, 50, 105, 111, 112, 25, -111, 12, -80],
      [-119, 113, 97, -127, 24, -21, -21, -58, 49, -75, 47, 64, 91, 48, -126, -83],
      [64, 27, -99, -73, 120, 86, -56, -32, -121, -2, -96, -98, -7, 19, -52, 76],
      [-77, 116, 87, -107, -32, 6, 19, 30, 50, 8, -62, 60, -122, -75, -32, -62],
      [-45, -104, -27, 27, -83, -7, 91, -13, 4, -11, 26, 81, 121, 81, 121, -9],
      [-114, -60, 87, -1, -63, -98, -119, -107, -26, 70, 69, -123, 79, -100, -27, -52],
      [-25, -25, -114, -110, -20, 2, -58, 48, -115, -9, 112, -52, 115, 46, -115, 81],
      [-15, -57, 102, -103, 14, -27, 91, 36, 61, 45, -31, -27, -105, 69, 101, 87],
      [-90, 6, -90, -70, -74, 44, -76, -3, 5, 82, -96, -87, -74, 89, -46, 107],
      [46, 16, -1, -25, 16, -29, -1, 16, -100, -67, 103, -103, -9, 126, 46, 4],
      [-110, 63, -91, -36, -43, -19, 1, 105, 16, 114, 78, -80, 57, -90, -54, 37],
      [42, 96, -41, 0, 66, -123, 92, -105, 2, -22, -67, 17, 105, -37, -76, -47]
  ]> : tensor<16x16xi8>
  %input_empty = tensor.empty() : tensor<8x16xf32>

  %input_q_empty = tensor.empty() : tensor<8x16xi8>
  %input_q = iree_linalg_ext.quantize_affine
      {indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>, affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>],
       quant_min = -128 : i64, quant_max = 127 : i64}
      ins(%input, %input_scale, %input_zp : tensor<8x16xf32>, f32, i64)
      outs(%input_q_empty : tensor<8x16xi8>) -> tensor<8x16xi8>
  %input_dq = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>, affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]}
      ins(%input_q, %input_scale, %input_zp : tensor<8x16xi8>, f32, i64)
      outs(%input_empty : tensor<8x16xf32>) -> tensor<8x16xf32>
  %weight_dq_empty = tensor.empty() : tensor<16x16xf32>
  %weight_dq = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(n, k) -> (n, k)>,
                        affine_map<(n, k) -> ()>, affine_map<(n, k) -> ()>,
                        affine_map<(n, k) -> (n, k)>]}
      ins(%weight, %weight_scale, %weight_zp : tensor<16x16xi8>, f32, i64)
      outs(%weight_dq_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%input_empty : tensor<8x16xf32>) -> tensor<8x16xf32>
  %linear = linalg.contract
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%input_dq, %weight_dq : tensor<8x16xf32>, tensor<16x16xf32>)
      outs(%init : tensor<8x16xf32>) -> tensor<8x16xf32>
  %output_q = iree_linalg_ext.quantize_affine
      {indexing_maps = [affine_map<(m, n) -> (m, n)>,
                        affine_map<(m, n) -> ()>, affine_map<(m, n) -> ()>,
                        affine_map<(m, n) -> (m, n)>],
       quant_min = -128 : i64, quant_max = 127 : i64}
      ins(%linear, %output_scale, %output_zp : tensor<8x16xf32>, f32, i64)
      outs(%input_q_empty : tensor<8x16xi8>) -> tensor<8x16xi8>
  %output = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(m, n) -> (m, n)>,
                        affine_map<(m, n) -> ()>, affine_map<(m, n) -> ()>,
                        affine_map<(m, n) -> (m, n)>]}
      ins(%output_q, %output_scale, %output_zp : tensor<8x16xi8>, f32, i64)
      outs(%input_empty : tensor<8x16xf32>) -> tensor<8x16xf32>

  // Recorded torch_quantized_output. The original IREE lowering matched
  // PyTorch exactly; require the same result with and without the QDQ rewrite.
  check.expect_eq_const(%output, dense<[
      [0.11787887662649155, 0.4929480254650116, 0.67512446641922, -0.7179895043373108, 0.13931140303611755, -1.2323700189590454, 0.16074392199516296, -0.43936672806739807, 0.06429757177829742, 0.5465293526649475, 0.11787887662649155, 0.7930033206939697, 0.2786228060722351, -0.03214878588914871, 0.6215431690216064, -0.3214878439903259],
      [0.2786228060722351, -0.2250414937734604, 0.18217644095420837, -0.6322594285011292, 0.4607992470264435, -0.18217644095420837, -0.7287057638168335, 0.6644082069396973, -0.010716261342167854, 0.7501382827758789, -0.7394220232963562, 0.02143252268433571, 0.2786228060722351, -0.15002766251564026, 0.09644635021686554, -0.6108269095420837],
      [-0.17146018147468567, -0.7930033206939697, -0.7715708017349243, -0.5786781311035156, 0.33220410346984863, 0.2143252193927765, -0.05358130484819412, 0.08573009073734283, 0.4500829875469208, 0.12859514355659485, -0.04286504536867142, -0.2893390655517578, -0.5143805742263794, -0.2786228060722351, -0.010716261342167854, 0.8787334561347961],
      [-0.4607992470264435, -0.6429756879806519, -0.5358130931854248, 0.7501382827758789, -0.6429756879806519, -0.07501383125782013, 0.5786781311035156, -0.7287057638168335, 0.13931140303611755, 1.103774905204773, 0.428650438785553, -0.6536919474601746, 0.2893390655517578, 0.06429757177829742, 0.2571902871131897, 0.9001659750938416],
      [-0.2357577532529831, -0.2143252193927765, -0.2679065465927124, -0.12859514355659485, 1.4681278467178345, -0.7179895043373108, 0.0, 0.06429757177829742, -0.010716261342167854, 0.15002766251564026, -1.0394773483276367, -0.2786228060722351, -1.2645188570022583, -0.4929480254650116, -0.07501383125782013, 0.20360895991325378],
      [0.5250968337059021, 0.17146018147468567, -0.08573009073734283, 0.2786228060722351, -0.6429756879806519, 0.2464740127325058, 0.2679065465927124, 0.8358683586120605, -0.921598494052887, -0.43936672806739807, -1.2323700189590454, -0.2893390655517578, -0.10716260969638824, -0.03214878588914871, 0.2893390655517578, 0.8787334561347961],
      [0.38578540086746216, 1.1895049810409546, 0.2679065465927124, 0.33220410346984863, 0.3107715845108032, -0.2679065465927124, -0.12859514355659485, 0.40721791982650757, -0.8037195801734924, -0.6322594285011292, -0.5572456121444702, 0.12859514355659485, -0.16074392199516296, 0.06429757177829742, 0.33220410346984863, -0.5893943905830383],
      [-0.15002766251564026, 0.15002766251564026, -0.4607992470264435, 0.67512446641922, -0.6215431690216064, -0.2893390655517578, 0.2250414937734604, -0.2357577532529831, -0.4607992470264435, -0.16074392199516296, -0.3107715845108032, -0.20360895991325378, -0.16074392199516296, 0.4929480254650116, -0.18217644095420837, 0.2464740127325058]
  ]> : tensor<8x16xf32>) : tensor<8x16xf32>
  return
}

// Execution counterparts of numerical corner cases in
// GlobalOptimization/test/convert_qdq_to_integer_math_scale_range.mlir.
// Tests ending in _reference or _rewritten document accepted numerical
// differences. The build targets use --gtest_filter to select their respective
// expectations; tests without these suffixes run in both configurations.
// Unfolded storage operands prevent constant folding of the contractions.

// (1 * 1e20) * (0 * 1e20) = 0. Sequential scaling avoids the overflowing
// scale product that previously caused the integer rewrite to return NaN.
func.func @scale_product_overflow() {
  %aq = util.unfoldable_constant dense<1> : tensor<1x1xi8>
  %bq = util.unfoldable_constant dense<0> : tensor<1x1xi8>
  %scale = arith.constant 1.0e20 : f32
  %ai = tensor.empty() : tensor<1x1xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]}
      ins(%aq, %scale : tensor<1x1xi8>, f32)
      outs(%ai : tensor<1x1xf32>) -> tensor<1x1xf32>
  %bi = tensor.empty() : tensor<1x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(k, n) -> (k, n)>,
                        affine_map<(k, n) -> ()>,
                        affine_map<(k, n) -> (k, n)>]}
      ins(%bq, %scale : tensor<1x1xi8>, f32)
      outs(%bi : tensor<1x1xf32>) -> tensor<1x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x1xf32>, tensor<1x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  check.expect_eq_const(%result, dense<0.0> : tensor<1x1xf32>) : tensor<1x1xf32>
  return
}

// f16 dequantization rounds 2049 to 2048, so -2048 + 2048 = 0. The integer
// rewrite skips that rounding and returns 1.
func.func private @compute_f16_dequantization_rounding() -> tensor<1x1xf16> {
  %aq = util.unfoldable_constant dense<[[-1, 1]]> : tensor<1x2xi8>
  %bq = util.unfoldable_constant dense<[[2048], [2049]]> : tensor<2x1xi16>
  %scale = arith.constant 1.0 : f32
  %ai = tensor.empty() : tensor<1x2xf16>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]}
      ins(%aq, %scale : tensor<1x2xi8>, f32)
      outs(%ai : tensor<1x2xf16>) -> tensor<1x2xf16>
  %bi = tensor.empty() : tensor<2x1xf16>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(k, n) -> (k, n)>,
                        affine_map<(k, n) -> ()>,
                        affine_map<(k, n) -> (k, n)>]}
      ins(%bq, %scale : tensor<2x1xi16>, f32)
      outs(%bi : tensor<2x1xf16>) -> tensor<2x1xf16>
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<1x1xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<1x1xf16>) -> tensor<1x1xf16>
  %result = linalg.matmul ins(%a, %b : tensor<1x2xf16>, tensor<2x1xf16>)
      outs(%init : tensor<1x1xf16>) -> tensor<1x1xf16>
  return %result : tensor<1x1xf16>
}

func.func @f16_dequantization_rounding_reference() {
  %result = call @compute_f16_dequantization_rounding() : () -> tensor<1x1xf16>
  check.expect_eq_const(%result, dense<0.0> : tensor<1x1xf16>) : tensor<1x1xf16>
  return
}

func.func @f16_dequantization_rounding_rewritten() {
  %result = call @compute_f16_dequantization_rounding() : () -> tensor<1x1xf16>
  check.expect_eq_const(%result, dense<1.0> : tensor<1x1xf16>) : tensor<1x1xf16>
  return
}

// -0 + (0 * -1) = -0. The integer rewrite discards the init sign and
// returns +0.
func.func private @compute_negative_zero_init() -> tensor<1x1xf32> {
  %aq = util.unfoldable_constant dense<0> : tensor<1x1xi8>
  %bq = util.unfoldable_constant dense<-1> : tensor<1x1xi8>
  %scale = arith.constant 1.0 : f32
  %ai = tensor.empty() : tensor<1x1xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]}
      ins(%aq, %scale : tensor<1x1xi8>, f32)
      outs(%ai : tensor<1x1xf32>) -> tensor<1x1xf32>
  %bi = tensor.empty() : tensor<1x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(k, n) -> (k, n)>,
                        affine_map<(k, n) -> ()>,
                        affine_map<(k, n) -> (k, n)>]}
      ins(%bq, %scale : tensor<1x1xi8>, f32)
      outs(%bi : tensor<1x1xf32>) -> tensor<1x1xf32>
  %zero = arith.constant -0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x1xf32>, tensor<1x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}

func.func @negative_zero_init_reference() {
  %result = call @compute_negative_zero_init() : () -> tensor<1x1xf32>
  // Floating-point equality cannot distinguish -0 from +0; check the bits.
  %c0 = arith.constant 0 : index
  %value = tensor.extract %result[%c0, %c0] : tensor<1x1xf32>
  %bits = arith.bitcast %value : f32 to i32
  %negative_zero_bits = arith.constant -2147483648 : i32
  %is_negative_zero = arith.cmpi eq, %bits, %negative_zero_bits : i32
  check.expect_true(%is_negative_zero) : i1
  return
}

func.func @negative_zero_init_rewritten() {
  %result = call @compute_negative_zero_init() : () -> tensor<1x1xf32>
  // Floating-point equality cannot distinguish -0 from +0; check the bits.
  %c0 = arith.constant 0 : index
  %value = tensor.extract %result[%c0, %c0] : tensor<1x1xf32>
  %bits = arith.bitcast %value : f32 to i32
  %positive_zero_bits = arith.constant 0 : i32
  %is_positive_zero = arith.cmpi eq, %bits, %positive_zero_bits : i32
  check.expect_true(%is_positive_zero) : i1
  return
}

// [0 * infinity, 1 * infinity] dot [1, 1] is NaN. The integer rewrite
// instead computes 1 * infinity and returns infinity.
func.func private @compute_infinite_scale_nan_propagation() -> tensor<1x1xf32> {
  %aq = util.unfoldable_constant dense<[[0, 1]]> : tensor<1x2xi8>
  %bq = util.unfoldable_constant dense<1> : tensor<2x1xi8>
  %sa = arith.constant 0x7F800000 : f32
  %sb = arith.constant 1.0 : f32
  %ai = tensor.empty() : tensor<1x2xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]}
      ins(%aq, %sa : tensor<1x2xi8>, f32)
      outs(%ai : tensor<1x2xf32>) -> tensor<1x2xf32>
  %bi = tensor.empty() : tensor<2x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(k, n) -> (k, n)>,
                        affine_map<(k, n) -> ()>,
                        affine_map<(k, n) -> (k, n)>]}
      ins(%bq, %sb : tensor<2x1xi8>, f32)
      outs(%bi : tensor<2x1xf32>) -> tensor<2x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x2xf32>, tensor<2x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}

func.func @infinite_scale_nan_propagation_reference() {
  %result = call @compute_infinite_scale_nan_propagation() : () -> tensor<1x1xf32>
  // Compare unordered rather than requiring a particular NaN payload.
  %c0 = arith.constant 0 : index
  %value = tensor.extract %result[%c0, %c0] : tensor<1x1xf32>
  %is_nan = arith.cmpf uno, %value, %value : f32
  check.expect_true(%is_nan) : i1
  return
}

func.func @infinite_scale_nan_propagation_rewritten() {
  %result = call @compute_infinite_scale_nan_propagation() : () -> tensor<1x1xf32>
  check.expect_eq_const(%result, dense<0x7F800000> : tensor<1x1xf32>) : tensor<1x1xf32>
  return
}

// (32767 * 1e-23)^2 is approximately 1.07368e-37. Sequential scaling avoids
// the underflowing scale product that previously erased the result.
func.func @scale_product_underflow() {
  %q = util.unfoldable_constant dense<32767> : tensor<1x1xi16>
  %scale = arith.constant 1.0e-23 : f32
  %ai = tensor.empty() : tensor<1x1xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]}
      ins(%q, %scale : tensor<1x1xi16>, f32)
      outs(%ai : tensor<1x1xf32>) -> tensor<1x1xf32>
  %bi = tensor.empty() : tensor<1x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(k, n) -> (k, n)>,
                        affine_map<(k, n) -> ()>,
                        affine_map<(k, n) -> (k, n)>]}
      ins(%q, %scale : tensor<1x1xi16>, f32)
      outs(%bi : tensor<1x1xf32>) -> tensor<1x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x1xf32>, tensor<1x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  // The dequantized operands and their product are all normal f32 values.
  // Use exact equality: an absolute tolerance could incorrectly accept zero.
  check.expect_eq_const(%result, dense<0x02122428> : tensor<1x1xf32>) : tensor<1x1xf32>
  return
}

// Both reference operands are finite, but D * 1e30 overflows in the epilogue.
func.func private @compute_opposing_scales_large_lhs() -> tensor<1x1xf32> {
  %aq = util.unfoldable_constant dense<-1> : tensor<1x32768xi8>
  %bq = util.unfoldable_constant dense<127> : tensor<32768x1xi8>
  %sa = arith.constant 1.0e30 : f32
  %sb = arith.constant 1.0e-30 : f32
  %ai = tensor.empty() : tensor<1x32768xf32>
  %a = iree_linalg_ext.dequantize_affine
      {input_unsigned, indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]}
      ins(%aq, %sa : tensor<1x32768xi8>, f32)
      outs(%ai : tensor<1x32768xf32>) -> tensor<1x32768xf32>
  %bi = tensor.empty() : tensor<32768x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(k, n) -> (k, n)>,
                        affine_map<(k, n) -> ()>,
                        affine_map<(k, n) -> (k, n)>]}
      ins(%bq, %sb : tensor<32768x1xi8>, f32)
      outs(%bi : tensor<32768x1xf32>) -> tensor<32768x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x32768xf32>, tensor<32768x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}

func.func @opposing_scales_large_lhs_reference() {
  %result = call @compute_opposing_scales_large_lhs() : () -> tensor<1x1xf32>
  // D = 32768 * 255 * 127 = 1061191680, within signed i32.
  // Allow reduction-order rounding while requiring a finite result.
  check.expect_almost_eq_const(%result, dense<1061191680.0> : tensor<1x1xf32>, atol 0.0, rtol 1.0e-4) : tensor<1x1xf32>
  return
}

func.func @opposing_scales_large_lhs_rewritten() {
  %result = call @compute_opposing_scales_large_lhs() : () -> tensor<1x1xf32>
  // 0x7F800000 -> INFINITY
  check.expect_eq_const(%result, dense<0x7F800000> : tensor<1x1xf32>) : tensor<1x1xf32>
  return
}

// Reversing the scales avoids epilogue overflow. Floating-point reduction
// rounding differs; allow 1e-4 relative error against the exact integer sum.
func.func private @compute_opposing_scales_large_rhs() -> tensor<1x1xf32> {
  %aq = util.unfoldable_constant dense<-1> : tensor<1x32768xi8>
  %bq = util.unfoldable_constant dense<127> : tensor<32768x1xi8>
  %sa = arith.constant 1.0e-30 : f32
  %sb = arith.constant 1.0e30 : f32
  %ai = tensor.empty() : tensor<1x32768xf32>
  %a = iree_linalg_ext.dequantize_affine
      {input_unsigned, indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]}
      ins(%aq, %sa : tensor<1x32768xi8>, f32)
      outs(%ai : tensor<1x32768xf32>) -> tensor<1x32768xf32>
  %bi = tensor.empty() : tensor<32768x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(k, n) -> (k, n)>,
                        affine_map<(k, n) -> ()>,
                        affine_map<(k, n) -> (k, n)>]}
      ins(%bq, %sb : tensor<32768x1xi8>, f32)
      outs(%bi : tensor<32768x1xf32>) -> tensor<32768x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x32768xf32>, tensor<32768x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}

func.func @opposing_scales_large_rhs_reference() {
  %result = call @compute_opposing_scales_large_rhs() : () -> tensor<1x1xf32>
  // D = 32768 * 255 * 127 = 1061191680, within signed i32.
  // Allow reduction-order rounding while requiring a finite result.
  check.expect_almost_eq_const(%result, dense<1061191680.0> : tensor<1x1xf32>, atol 0.0, rtol 1.0e-4) : tensor<1x1xf32>
  return
}

func.func @opposing_scales_large_rhs_rewritten() {
  %result = call @compute_opposing_scales_large_rhs() : () -> tensor<1x1xf32>
  check.expect_eq_const(%result, dense<1061191680.0> : tensor<1x1xf32>) : tensor<1x1xf32>
  return
}

// All scales are finite, but 127 * 1e37 overflows during reference
// dequantization. Integer reduction bypasses that overflow.
func.func private @compute_finite_scale_dequantization_overflow() -> tensor<1x1xf32> {
  %aq = util.unfoldable_constant dense<-1> : tensor<1x32768xi8>
  %bq = util.unfoldable_constant dense<127> : tensor<32768x1xi8>
  %sa = arith.constant 1.0e-37 : f32
  %sb = arith.constant 1.0e37 : f32
  %ai = tensor.empty() : tensor<1x32768xf32>
  %a = iree_linalg_ext.dequantize_affine
      {input_unsigned, indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]}
      ins(%aq, %sa : tensor<1x32768xi8>, f32)
      outs(%ai : tensor<1x32768xf32>) -> tensor<1x32768xf32>
  %bi = tensor.empty() : tensor<32768x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(k, n) -> (k, n)>,
                        affine_map<(k, n) -> ()>,
                        affine_map<(k, n) -> (k, n)>]}
      ins(%bq, %sb : tensor<32768x1xi8>, f32)
      outs(%bi : tensor<32768x1xf32>) -> tensor<32768x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x32768xf32>, tensor<32768x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}

func.func @finite_scale_dequantization_overflow_reference() {
  %result = call @compute_finite_scale_dequantization_overflow() : () -> tensor<1x1xf32>
  check.expect_eq_const(%result, dense<0x7F800000> : tensor<1x1xf32>) : tensor<1x1xf32>
  return
}

func.func @finite_scale_dequantization_overflow_rewritten() {
  %result = call @compute_finite_scale_dequantization_overflow() : () -> tensor<1x1xf32>
  check.expect_eq_const(%result, dense<1061191616.0> : tensor<1x1xf32>) : tensor<1x1xf32>
  return
}

// With zero activations, the same overflowing reference weights yield NaN
// from 0 * infinity. Integer reduction instead yields zero.
func.func private @compute_finite_scale_dequantization_nan() -> tensor<1x1xf32> {
  %aq = util.unfoldable_constant dense<0> : tensor<1x32768xi8>
  %bq = util.unfoldable_constant dense<127> : tensor<32768x1xi8>
  %sa = arith.constant 1.0e-37 : f32
  %sb = arith.constant 1.0e37 : f32
  %ai = tensor.empty() : tensor<1x32768xf32>
  %a = iree_linalg_ext.dequantize_affine
      {input_unsigned, indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]}
      ins(%aq, %sa : tensor<1x32768xi8>, f32)
      outs(%ai : tensor<1x32768xf32>) -> tensor<1x32768xf32>
  %bi = tensor.empty() : tensor<32768x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(k, n) -> (k, n)>,
                        affine_map<(k, n) -> ()>,
                        affine_map<(k, n) -> (k, n)>]}
      ins(%bq, %sb : tensor<32768x1xi8>, f32)
      outs(%bi : tensor<32768x1xf32>) -> tensor<32768x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x32768xf32>, tensor<32768x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}

func.func @finite_scale_dequantization_nan_reference() {
  %result = call @compute_finite_scale_dequantization_nan() : () -> tensor<1x1xf32>
  // Compare unordered without requiring a specific NaN payload.
  %c0 = arith.constant 0 : index
  %value = tensor.extract %result[%c0, %c0] : tensor<1x1xf32>
  %is_nan = arith.cmpf uno, %value, %value : f32
  check.expect_true(%is_nan) : i1
  return
}

func.func @finite_scale_dequantization_nan_rewritten() {
  %result = call @compute_finite_scale_dequantization_nan() : () -> tensor<1x1xf32>
  check.expect_eq_const(%result, dense<0.0> : tensor<1x1xf32>) : tensor<1x1xf32>
  return
}
