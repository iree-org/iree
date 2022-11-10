// Test large aligned linalg matmul to make sure we go through the optimized
// path for GPUs.

// Problem size      : 2048x512x1024
// Input type        : F32
// Accumulation type : F32
func.func @matmul_2048x512x1024_f32_f32() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<2048x1024xf32>
  %rhs = util.unfoldable_constant dense<0.4> : tensor<1024x512xf32>
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<2048x512xf32>
  %CC = linalg.fill ins(%c0 : f32) outs(%init : tensor<2048x512xf32>) -> tensor<2048x512xf32>
  %D = linalg.matmul ins(%lhs, %rhs: tensor<2048x1024xf32>, tensor<1024x512xf32>)
                    outs(%CC: tensor<2048x512xf32>) -> tensor<2048x512xf32>
  check.expect_almost_eq_const(%D, dense<409.596> : tensor<2048x512xf32>) : tensor<2048x512xf32>
  return
}

// Problem size      : 3456x1024x2048
// Input type        : F16
// Accumulation type : F16
func.func @matmul_3456x1024x2048_f16_f16() {
  %lhs = util.unfoldable_constant dense<1.00> : tensor<3456x2048xf16>
  %rhs = util.unfoldable_constant dense<0.01> : tensor<2048x1024xf16>
  %c0 = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<3456x1024xf16>
  %CC = linalg.fill ins(%c0 : f16) outs(%init : tensor<3456x1024xf16>) -> tensor<3456x1024xf16>
  %D = linalg.matmul ins(%lhs, %rhs: tensor<3456x2048xf16>, tensor<2048x1024xf16>)
                    outs(%CC: tensor<3456x1024xf16>) -> tensor<3456x1024xf16>
  check.expect_almost_eq_const(%D, dense<20.2812> : tensor<3456x1024xf16>) : tensor<3456x1024xf16>
  return
}