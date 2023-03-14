func.func @winograd_output() {
  %input = util.unfoldable_constant dense<1.0> : tensor<8x8x1x1x1x1xf32>

  %init = tensor.empty() : tensor<1x6x6x1xf32>
  %1 = iree_linalg_ext.winograd.output_transform
       output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
       ins(%input : tensor<8x8x1x1x1x1xf32>)
       outs(%init : tensor<1x6x6x1xf32>) -> tensor<1x6x6x1xf32>
  %2 = flow.tensor.reshape %1 : tensor<1x6x6x1xf32> -> tensor<6x6xf32>

  check.expect_almost_eq_const(
      %2,
      dense<[[   49.00000,     0.00000,    73.50000,     0.00000,   238.87500,     7.00000],
             [    0.00000,     0.00000,     0.00000,     0.00000,     0.00000,     0.00000],
             [   73.50000,     0.00000,   110.25000,     0.00000,   358.31250,    10.50000],
             [    0.00000,     0.00000,     0.00000,     0.00000,     0.00000,     0.00000],
             [  238.87500,     0.00000,   358.31250,     0.00000,  1164.51562,    34.12500],
             [    7.00000,     0.00000,    10.50000,     0.00000,    34.12500,     1.00000]]> : tensor<6x6xf32>
  ) : tensor<6x6xf32>

  return
}

func.func @winograd_output_nchw() {
  %input = util.unfoldable_constant dense<1.0> : tensor<8x8x1x1x1x1xf32>

  %init = tensor.empty() : tensor<1x1x6x6xf32>
  %1 = iree_linalg_ext.winograd.output_transform
       output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
       ins(%input : tensor<8x8x1x1x1x1xf32>)
       outs(%init : tensor<1x1x6x6xf32>) -> tensor<1x1x6x6xf32>
  %2 = flow.tensor.reshape %1 : tensor<1x1x6x6xf32> -> tensor<6x6xf32>

  check.expect_almost_eq_const(
      %2,
      dense<[[   49.00000,     0.00000,    73.50000,     0.00000,   238.87500,     7.00000],
             [    0.00000,     0.00000,     0.00000,     0.00000,     0.00000,     0.00000],
             [   73.50000,     0.00000,   110.25000,     0.00000,   358.31250,    10.50000],
             [    0.00000,     0.00000,     0.00000,     0.00000,     0.00000,     0.00000],
             [  238.87500,     0.00000,   358.31250,     0.00000,  1164.51562,    34.12500],
             [    7.00000,     0.00000,    10.50000,     0.00000,    34.12500,     1.00000]]> : tensor<6x6xf32>
  ) : tensor<6x6xf32>

  return
}
