// linalg.matvec: constant matrix [4096, 4096] times vector [4096], f32.

util.global private @weight = #util.byte_pattern<63> : tensor<4096x4096xf32>

func.func @matvec_4096x4096_f32(%x: tensor<4096xf32>) -> tensor<4096xf32> {
  %weight = util.global.load @weight : tensor<4096x4096xf32>
  %e1 = tensor.empty() : tensor<4096xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<4096xf32>) -> tensor<4096xf32>
  %mv4 = linalg.matvec ins(%weight, %x : tensor<4096x4096xf32>, tensor<4096xf32>) outs(%f3 : tensor<4096xf32>) -> tensor<4096xf32>
  return %mv4 : tensor<4096xf32>
}
