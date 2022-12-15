func.func @winograd_input() {
  %input = util.unfoldable_constant dense<1.0> : tensor<1x6x6x1xf32>

  %init = tensor.empty() : tensor<8x8x1x1x1x1xf32>
  %1 = iree_linalg_ext.winograd.input_transform
       output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
       ins(%input : tensor<1x6x6x1xf32>)
       outs(%init : tensor<8x8x1x1x1x1xf32>) -> tensor<8x8x1x1x1x1xf32>
  %2 = flow.tensor.reshape %1 : tensor<8x8x1x1x1x1xf32> -> tensor<8x8xf32>

  check.expect_almost_eq_const(
      %2,
      dense<[[ 1.0000, -5.5000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
             [-5.5000, 30.2500,  5.5000,  5.5000,  5.5000,  5.5000,  5.5000,  5.5000],
             [-1.0000,  5.5000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
             [-1.0000,  5.5000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
             [-1.0000,  5.5000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
             [-1.0000,  5.5000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
             [-1.0000,  5.5000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
             [-1.0000,  5.5000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000]]> : tensor<8x8xf32>
  ) : tensor<8x8xf32>

  return
}

func.func @winograd_input_nchw() {
  %input = util.unfoldable_constant dense<1.0> : tensor<1x1x6x6xf32>

  %init = tensor.empty() : tensor<8x8x1x1x1x1xf32>
  %1 = iree_linalg_ext.winograd.input_transform
       output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
       ins(%input : tensor<1x1x6x6xf32>)
       outs(%init : tensor<8x8x1x1x1x1xf32>) -> tensor<8x8x1x1x1x1xf32>
  %2 = flow.tensor.reshape %1 : tensor<8x8x1x1x1x1xf32> -> tensor<8x8xf32>

  check.expect_almost_eq_const(
      %2,
      dense<[[ 1.0000, -5.5000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
             [-5.5000, 30.2500,  5.5000,  5.5000,  5.5000,  5.5000,  5.5000,  5.5000],
             [-1.0000,  5.5000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
             [-1.0000,  5.5000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
             [-1.0000,  5.5000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
             [-1.0000,  5.5000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
             [-1.0000,  5.5000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
             [-1.0000,  5.5000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000]]> : tensor<8x8xf32>
  ) : tensor<8x8xf32>

  return
}
