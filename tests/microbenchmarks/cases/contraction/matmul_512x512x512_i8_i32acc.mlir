// Matmul MxNxK = 512x512x512, i8 operands, i32 accumulator.

func.func @matmul_512x512x512_i8_i32acc(%a: tensor<512x512xi8>, %b: tensor<512x512xi8>) -> tensor<512x512xi32> {
  %e1 = tensor.empty() : tensor<512x512xi32>
  %c2 = arith.constant 0 : i32
  %f3 = linalg.fill ins(%c2 : i32) outs(%e1 : tensor<512x512xi32>) -> tensor<512x512xi32>
  %mm4 = linalg.matmul ins(%a, %b : tensor<512x512xi8>, tensor<512x512xi8>) outs(%f3 : tensor<512x512xi32>) -> tensor<512x512xi32>
  return %mm4 : tensor<512x512xi32>
}
