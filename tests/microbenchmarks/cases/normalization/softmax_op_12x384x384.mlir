// linalg.softmax over the last dimension, 12x384x384 f32.

func.func @softmax_op_12x384x384(%x: tensor<12x384x384xf32>) -> tensor<12x384x384xf32> {
  %e = tensor.empty() : tensor<12x384x384xf32>
  %r = linalg.softmax dimension(2) ins(%x : tensor<12x384x384xf32>) outs(%e : tensor<12x384x384xf32>) -> tensor<12x384x384xf32>
  return %r : tensor<12x384x384xf32>
}
