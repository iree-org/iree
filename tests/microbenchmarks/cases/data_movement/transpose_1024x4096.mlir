// 2-D transpose, 1024x4096 -> 4096x1024 f32.

func.func @transpose_1024x4096(%x: tensor<1024x4096xf32>) -> tensor<4096x1024xf32> {
  %e = tensor.empty() : tensor<4096x1024xf32>
  %r = linalg.transpose ins(%x : tensor<1024x4096xf32>) outs(%e : tensor<4096x1024xf32>) permutation = [1, 0]
  return %r : tensor<4096x1024xf32>
}
