//===----------------------------------------------------------------------===//
// Linalg matmul ops.
//===----------------------------------------------------------------------===//

func @matmul_384x384x512() -> tensor<384x512xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<384x384xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<384x512xf32>
    %dst = util.unfoldable_constant dense<1.0> : tensor<384x512xf32>
    %0 = linalg.matmul ins(%lhs, %rhs : tensor<384x384xf32>, tensor<384x512xf32>) outs(%dst : tensor<384x512xf32>) -> tensor<384x512xf32>
    return %0 : tensor<384x512xf32>
}

func @mmt4d_384x384x512() -> tensor<96x128x4x4xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<96x96x4x4xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<128x96x4x4xf32>
    %dst = util.unfoldable_constant dense<1.0> : tensor<96x128x4x4xf32>
    %0 = linalg.mmt4d ins(%lhs, %rhs : tensor<96x96x4x4xf32>, tensor<128x96x4x4xf32>) outs(%dst : tensor<96x128x4x4xf32>) -> tensor<96x128x4x4xf32>
    return %0 : tensor<96x128x4x4xf32>
}
