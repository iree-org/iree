// Zero padding by 1 on both spatial dimensions, NCHW, 1x64x56x56 -> 1x64x58x58 f32.

func.func @pad_nchw_1x64x56x56(%x: tensor<1x64x56x56xf32>) -> tensor<1x64x58x58xf32> {
  %zero = arith.constant 0.0 : f32
  %p = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
    tensor.yield %zero : f32
  } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
  return %p : tensor<1x64x58x58xf32>
}
