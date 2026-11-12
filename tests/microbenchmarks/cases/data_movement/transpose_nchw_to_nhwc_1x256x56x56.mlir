// Layout transpose NCHW -> NHWC, 1x256x56x56 f32.

func.func @transpose_nchw_to_nhwc_1x256x56x56(%x: tensor<1x256x56x56xf32>) -> tensor<1x56x56x256xf32> {
  %e = tensor.empty() : tensor<1x56x56x256xf32>
  %r = linalg.transpose ins(%x : tensor<1x256x56x56xf32>) outs(%e : tensor<1x56x56x256xf32>) permutation = [0, 2, 3, 1]
  return %r : tensor<1x56x56x256xf32>
}
