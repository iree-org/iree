// Linear layer: activation [384, 768] times constant weight [768, 768], f32.

util.global private @weight = #util.byte_pattern<63> : tensor<768x768xf32>

func.func @linear_384x768x768(%x: tensor<384x768xf32>) -> tensor<384x768xf32> {
  %weight = util.global.load @weight : tensor<768x768xf32>
  %e1 = tensor.empty() : tensor<384x768xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<384x768xf32>) -> tensor<384x768xf32>
  %mm4 = linalg.matmul ins(%x, %weight : tensor<384x768xf32>, tensor<768x768xf32>) outs(%f3 : tensor<384x768xf32>) -> tensor<384x768xf32>
  return %mm4 : tensor<384x768xf32>
}
