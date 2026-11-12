// Linear layer with M = 1 (decode-like GEMV through linalg.matmul), f32,
// constant weight [4096, 4096].

util.global private @weight = #util.byte_pattern<63> : tensor<4096x4096xf32>

func.func @linear_1x4096x4096(%x: tensor<1x4096xf32>) -> tensor<1x4096xf32> {
  %weight = util.global.load @weight : tensor<4096x4096xf32>
  %e1 = tensor.empty() : tensor<1x4096xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
  %mm4 = linalg.matmul ins(%x, %weight : tensor<1x4096xf32>, tensor<4096x4096xf32>) outs(%f3 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
  return %mm4 : tensor<1x4096xf32>
}
