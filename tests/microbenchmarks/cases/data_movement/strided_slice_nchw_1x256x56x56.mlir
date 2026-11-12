// Strided spatial subsampling with tensor.extract_slice (stride 2), NCHW,
// 1x256x56x56 -> 1x256x28x28 f32. This is the data movement a 1x1 stride-2 conv
// turns into when its stride is folded into the input.

func.func @strided_slice_nchw_1x256x56x56(%x: tensor<1x256x56x56xf32>) -> tensor<1x256x28x28xf32> {
  %r = tensor.extract_slice %x[0, 0, 0, 0] [1, 256, 28, 28] [1, 1, 2, 2] : tensor<1x256x56x56xf32> to tensor<1x256x28x28xf32>
  return %r : tensor<1x256x28x28xf32>
}
