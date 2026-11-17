// Runs with the integer rewrite enabled and disabled. The expected values come
// from the PT2E `quantized_decomposed` ops in PyTorch.

// choose_qparams.tensor, per row.
func.func @asymmetric() {
  %x = util.unfoldable_constant dense<[[-1.5, 0.25, 2.0, 3.5], [0.1, 0.2, 0.3, 0.4], [-4.0, -2.0, -1.0, -0.5], [0.0, 0.0, 0.0, 0.0]]> : tensor<4x4xf32>
  %si = tensor.empty() : tensor<4xf32>
  %zi = tensor.empty() : tensor<4xi32>
  %r:2 = iree_linalg_ext.choose_qparams_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>],
       mapping = asymmetric, quant_min = -128, quant_max = 127, eps = 1.0e-05>
      ins(%x : tensor<4x4xf32>) outs(%si, %zi : tensor<4xf32>, tensor<4xi32>) -> tensor<4xf32>, tensor<4xi32>
  check.expect_almost_eq_const(%r#0, dense<[1.9607844e-02, 1.5686274e-03, 1.5686275e-02, 9.9999997e-06]> : tensor<4xf32>, atol 0.0, rtol 1.0e-6) : tensor<4xf32>
  check.expect_eq_const(%r#1, dense<[-52, -128, 127, -128]> : tensor<4xi32>) : tensor<4xi32>
  return
}

// choose_qparams_per_token_asymmetric, which uses the f32 machine epsilon. The
// zero point of the all-zero row comes from the upper bound.
func.func @asymmetric_min_error() {
  %x = util.unfoldable_constant dense<[[-1.5, 0.25, 2.0, 3.5], [0.1, 0.2, 0.3, 0.4], [-4.0, -2.0, -1.0, -0.5], [0.0, 0.0, 0.0, 0.0]]> : tensor<4x4xf32>
  %si = tensor.empty() : tensor<4xf32>
  %zi = tensor.empty() : tensor<4xi32>
  %r:2 = iree_linalg_ext.choose_qparams_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>],
       mapping = asymmetric_min_error, quant_min = -128, quant_max = 127, eps = 1.1920928955078125e-07>
      ins(%x : tensor<4x4xf32>) outs(%si, %zi : tensor<4xf32>, tensor<4xi32>) -> tensor<4xf32>, tensor<4xi32>
  check.expect_almost_eq_const(%r#0, dense<[1.9607844e-02, 1.5686274e-03, 1.5686275e-02, 1.1920929e-07]> : tensor<4xf32>, atol 0.0, rtol 1.0e-6) : tensor<4xf32>
  check.expect_eq_const(%r#1, dense<[-52, -128, 127, 127]> : tensor<4xi32>) : tensor<4xi32>
  return
}

// choose_qparams_symmetric.tensor, per row.
func.func @symmetric() {
  %x = util.unfoldable_constant dense<[[-1.5, 0.25, 2.0, 3.5], [0.1, 0.2, 0.3, 0.4], [-4.0, -2.0, -1.0, -0.5], [0.0, 0.0, 0.0, 0.0]]> : tensor<4x4xf32>
  %si = tensor.empty() : tensor<4xf32>
  %zi = tensor.empty() : tensor<4xi32>
  %r:2 = iree_linalg_ext.choose_qparams_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>],
       mapping = symmetric, quant_min = -128, quant_max = 127, eps = 1.0e-05>
      ins(%x : tensor<4x4xf32>) outs(%si, %zi : tensor<4xf32>, tensor<4xi32>) -> tensor<4xf32>, tensor<4xi32>
  check.expect_almost_eq_const(%r#0, dense<[2.7450981e-02, 3.1372549e-03, 3.1372551e-02, 9.9999997e-06]> : tensor<4xf32>, atol 0.0, rtol 1.0e-6) : tensor<4xf32>
  check.expect_eq_const(%r#1, dense<[0, 0, 0, 0]> : tensor<4xi32>) : tensor<4xi32>
  return
}

// choose_qparams_per_token, without a zero point.
func.func @symmetric_qmax() {
  %x = util.unfoldable_constant dense<[[-1.5, 0.25, 2.0, 3.5], [0.1, 0.2, 0.3, 0.4], [-4.0, -2.0, -1.0, -0.5], [0.0, 0.0, 0.0, 0.0]]> : tensor<4x4xf32>
  %si = tensor.empty() : tensor<4xf32>
  %r = iree_linalg_ext.choose_qparams_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>],
       mapping = symmetric_qmax, quant_min = -128, quant_max = 127, eps = 1.0e-05>
      ins(%x : tensor<4x4xf32>) outs(%si : tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%r, dense<[2.7559055e-02, 3.1496063e-03, 3.1496063e-02, 7.8740158e-08]> : tensor<4xf32>, atol 0.0, rtol 1.0e-6) : tensor<4xf32>
  return
}

// Dynamic per-token quantization of the activations of a linear layer with
// per-channel int8 weights. The parameters are only known at runtime, and the
// integer rewrite still applies to the contraction.
func.func @dynamic_per_token_linear() {
  %x = util.unfoldable_constant dense<[[-1.5, 0.25, 2.0, 3.5], [0.1, 0.2, 0.3, 0.4], [-4.0, -2.0, -1.0, -0.5], [0.0, 0.0, 0.0, 0.0]]> : tensor<4x4xf32>
  %wq = util.unfoldable_constant dense<[[1, -2, 3], [4, 5, -6], [-7, 8, 9], [10, -11, 12]]> : tensor<4x3xi8>
  %ws = util.unfoldable_constant dense<[0.5, 0.25, 0.125]> : tensor<3xf32>
  %si = tensor.empty() : tensor<4xf32>
  %zi = tensor.empty() : tensor<4xi8>
  %p:2 = iree_linalg_ext.choose_qparams_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>],
       mapping = asymmetric_min_error, quant_min = -128, quant_max = 127,
       eps = 1.1920928955078125e-07>
      ins(%x : tensor<4x4xf32>)
      outs(%si, %zi : tensor<4xf32>, tensor<4xi8>) -> tensor<4xf32>, tensor<4xi8>
  %qi = tensor.empty() : tensor<4x4xi8>
  %q = iree_linalg_ext.quantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       quant_min = -128, quant_max = 127>
      ins(%x, %p#0, %p#1 : tensor<4x4xf32>, tensor<4xf32>, tensor<4xi8>)
      outs(%qi : tensor<4x4xi8>) -> tensor<4x4xi8>
  %ai = tensor.empty() : tensor<4x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%q, %p#0, %p#1 : tensor<4x4xi8>, tensor<4xf32>, tensor<4xi8>)
      outs(%ai : tensor<4x4xf32>) -> tensor<4x4xf32>
  %bi = tensor.empty() : tensor<4x3xf32>
  %b = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%wq, %ws : tensor<4x3xi8>, tensor<3xf32>)
      outs(%bi : tensor<4x3xf32>) -> tensor<4x3xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x3xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<4x3xf32>) -> tensor<4x3xf32>
  %r = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x3xf32>)
      outs(%init : tensor<4x3xf32>) -> tensor<4x3xf32>
  check.expect_almost_eq_const(%r, dense<[[10.215687, -4.534314, 6.7352943], [1.4031372, -0.30000007, 0.82411766], [-4.9803925, -1.1176473, -1.8882353], [0.0, 0.0, 0.0]]> : tensor<4x3xf32>, atol 1.0e-5, rtol 1.0e-5) : tensor<4x3xf32>
  return
}

// choose_qparams.tensor over the whole tensor: min = -4.0, max = 3.5.
func.func @asymmetric_per_tensor() {
  %x = util.unfoldable_constant dense<[[-1.5, 0.25, 2.0, 3.5], [0.1, 0.2, 0.3, 0.4], [-4.0, -2.0, -1.0, -0.5], [0.0, 0.0, 0.0, 0.0]]> : tensor<4x4xf32>
  %si = tensor.empty() : tensor<f32>
  %zi = tensor.empty() : tensor<i32>
  %r:2 = iree_linalg_ext.choose_qparams_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>],
       mapping = asymmetric, quant_min = -128, quant_max = 127, eps = 1.0e-05>
      ins(%x : tensor<4x4xf32>) outs(%si, %zi : tensor<f32>, tensor<i32>) -> tensor<f32>, tensor<i32>
  check.expect_almost_eq_const(%r#0, dense<2.9411765e-02> : tensor<f32>, atol 0.0, rtol 1.0e-6) : tensor<f32>
  check.expect_eq_const(%r#1, dense<8> : tensor<i32>) : tensor<i32>
  return
}

// choose_qparams_symmetric.tensor over the whole tensor.
func.func @symmetric_per_tensor() {
  %x = util.unfoldable_constant dense<[[-1.5, 0.25, 2.0, 3.5], [0.1, 0.2, 0.3, 0.4], [-4.0, -2.0, -1.0, -0.5], [0.0, 0.0, 0.0, 0.0]]> : tensor<4x4xf32>
  %si = tensor.empty() : tensor<f32>
  %zi = tensor.empty() : tensor<i32>
  %r:2 = iree_linalg_ext.choose_qparams_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>],
       mapping = symmetric, quant_min = -128, quant_max = 127, eps = 1.0e-05>
      ins(%x : tensor<4x4xf32>) outs(%si, %zi : tensor<f32>, tensor<i32>) -> tensor<f32>, tensor<i32>
  check.expect_almost_eq_const(%r#0, dense<3.1372551e-02> : tensor<f32>, atol 0.0, rtol 1.0e-6) : tensor<f32>
  check.expect_eq_const(%r#1, dense<0> : tensor<i32>) : tensor<i32>
  return
}

// choose_qparams_per_token_asymmetric on a rank-3 input, with one token per
// pair of leading indices.
func.func @asymmetric_min_error_per_token_rank3() {
  %x = util.unfoldable_constant dense<[[[-1.5, 0.25, 2.0, 3.5], [0.1, 0.2, 0.3, 0.4]], [[-4.0, -2.0, -1.0, -0.5], [0.0, 0.0, 0.0, 0.0]]]> : tensor<2x2x4xf32>
  %si = tensor.empty() : tensor<2x2xf32>
  %zi = tensor.empty() : tensor<2x2xi32>
  %r:2 = iree_linalg_ext.choose_qparams_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>],
       mapping = asymmetric_min_error, quant_min = -128, quant_max = 127, eps = 1.1920928955078125e-07>
      ins(%x : tensor<2x2x4xf32>) outs(%si, %zi : tensor<2x2xf32>, tensor<2x2xi32>) -> tensor<2x2xf32>, tensor<2x2xi32>
  check.expect_almost_eq_const(%r#0, dense<[[1.9607844e-02, 1.5686274e-03], [1.5686275e-02, 1.1920929e-07]]> : tensor<2x2xf32>, atol 0.0, rtol 1.0e-6) : tensor<2x2xf32>
  check.expect_eq_const(%r#1, dense<[[-52, -128], [127, 127]]> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

// choose_qparams_per_token with a dynamic number of tokens.
func.func @symmetric_qmax_dynamic() {
  %x = flow.tensor.dynamic_constant dense<[[-1.5, 0.25, 2.0, 3.5], [0.1, 0.2, 0.3, 0.4], [-4.0, -2.0, -1.0, -0.5], [0.0, 0.0, 0.0, 0.0]]> : tensor<4x4xf32> -> tensor<?x4xf32>
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %x, %c0 : tensor<?x4xf32>
  %si = tensor.empty(%d0) : tensor<?xf32>
  %r = iree_linalg_ext.choose_qparams_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>],
       mapping = symmetric_qmax, quant_min = -128, quant_max = 127, eps = 1.0e-05>
      ins(%x : tensor<?x4xf32>) outs(%si : tensor<?xf32>) -> tensor<?xf32>
  %s = tensor.cast %r : tensor<?xf32> to tensor<4xf32>
  check.expect_almost_eq_const(%s, dense<[2.7559055e-02, 3.1496063e-03, 3.1496063e-02, 7.8740158e-08]> : tensor<4xf32>, atol 0.0, rtol 1.0e-6) : tensor<4xf32>
  return
}

// choose_qparams.tensor with a dynamic reduced dimension.
func.func @asymmetric_per_tensor_dynamic() {
  %x = flow.tensor.dynamic_constant dense<[[-1.5, 0.25, 2.0, 3.5], [0.1, 0.2, 0.3, 0.4], [-4.0, -2.0, -1.0, -0.5], [0.0, 0.0, 0.0, 0.0]]> : tensor<4x4xf32> -> tensor<?x?xf32>
  %si = tensor.empty() : tensor<f32>
  %zi = tensor.empty() : tensor<i32>
  %r:2 = iree_linalg_ext.choose_qparams_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>],
       mapping = asymmetric, quant_min = -128, quant_max = 127, eps = 1.0e-05>
      ins(%x : tensor<?x?xf32>) outs(%si, %zi : tensor<f32>, tensor<i32>) -> tensor<f32>, tensor<i32>
  check.expect_almost_eq_const(%r#0, dense<2.9411765e-02> : tensor<f32>, atol 0.0, rtol 1.0e-6) : tensor<f32>
  check.expect_eq_const(%r#1, dense<8> : tensor<i32>) : tensor<i32>
  return
}
