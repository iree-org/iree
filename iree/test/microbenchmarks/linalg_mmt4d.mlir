//===----------------------------------------------------------------------===//
// Linalg matmul ops.
//===----------------------------------------------------------------------===//

func.func @matmul_384x384x512() -> tensor<384x512xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<384x384xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<384x512xf32>
    %dst = util.unfoldable_constant dense<1.0> : tensor<384x512xf32>
    %0 = linalg.matmul ins(%lhs, %rhs : tensor<384x384xf32>, tensor<384x512xf32>) outs(%dst : tensor<384x512xf32>) -> tensor<384x512xf32>
    return %0 : tensor<384x512xf32>
}

func.func @mmt4d_384x384x512_4x1x4() -> tensor<96x128x4x4xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<96x384x4x1xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<128x384x4x1xf32>
    %dst = util.unfoldable_constant dense<1.0> : tensor<96x128x4x4xf32>
    %0 = linalg.mmt4d ins(%lhs, %rhs : tensor<96x384x4x1xf32>, tensor<128x384x4x1xf32>) outs(%dst : tensor<96x128x4x4xf32>) -> tensor<96x128x4x4xf32>
    return %0 : tensor<96x128x4x4xf32>
}

func.func @mmt4d_384x384x512_8x1x8() -> tensor<48x64x8x8xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<48x384x8x1xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<64x384x8x1xf32>
    %dst = util.unfoldable_constant dense<1.0> : tensor<48x64x8x8xf32>
    %0 = linalg.mmt4d ins(%lhs, %rhs : tensor<48x384x8x1xf32>, tensor<64x384x8x1xf32>) outs(%dst : tensor<48x64x8x8xf32>) -> tensor<48x64x8x8xf32>
    return %0 : tensor<48x64x8x8xf32>
}
