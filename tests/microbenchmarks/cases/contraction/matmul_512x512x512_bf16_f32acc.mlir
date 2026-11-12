// Matmul MxNxK = 512x512x512, bf16 operands, f32 accumulator.

func.func @matmul_512x512x512_bf16_f32acc(%a: tensor<512x512xbf16>, %b: tensor<512x512xbf16>) -> tensor<512x512xf32> {
  %e1 = tensor.empty() : tensor<512x512xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<512x512xf32>) -> tensor<512x512xf32>
  %mm4 = linalg.matmul ins(%a, %b : tensor<512x512xbf16>, tensor<512x512xbf16>) outs(%f3 : tensor<512x512xf32>) -> tensor<512x512xf32>
  return %mm4 : tensor<512x512xf32>
}
