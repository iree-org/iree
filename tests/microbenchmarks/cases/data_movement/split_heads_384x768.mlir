// Attention head split: expand [384, 768] to [384, 12, 64] and transpose to
// [12, 384, 64], f32.

func.func @split_heads_384x768(%x: tensor<384x768xf32>) -> tensor<12x384x64xf32> {
  %ex = tensor.expand_shape %x [[0], [1, 2]] output_shape [384, 12, 64] : tensor<384x768xf32> into tensor<384x12x64xf32>
  %e = tensor.empty() : tensor<12x384x64xf32>
  %r = linalg.transpose ins(%ex : tensor<384x12x64xf32>) outs(%e : tensor<12x384x64xf32>) permutation = [1, 0, 2]
  return %r : tensor<12x384x64xf32>
}
