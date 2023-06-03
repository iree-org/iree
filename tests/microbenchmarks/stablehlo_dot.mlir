//===----------------------------------------------------------------------===//
// O(N^3) matmul ops.
//===----------------------------------------------------------------------===//

func.func @dot_384x384x512() -> tensor<384x512xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<384x384xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<384x512xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<384x384xf32>, tensor<384x512xf32>) -> tensor<384x512xf32>
    return %0: tensor<384x512xf32>
}

func.func @dot_384x128x128() -> tensor<384x128xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<384x128xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<128x128xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    return %0 : tensor<384x128xf32>
}

func.func @dot_384x128x512() -> tensor<384x512xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<384x128xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<128x512xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    return %0 : tensor<384x512xf32>
}

func.func @dot_384x512x128() -> tensor<384x128xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<384x512xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<512x128xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    return %0 : tensor<384x128xf32>
}

func.func @dot_384x512x2() -> tensor<384x2xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<384x512xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<512x2xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<384x512xf32>, tensor<512x2xf32>) -> tensor<384x2xf32>
    return %0 : tensor<384x2xf32>
}

func.func @dot_384x384x32() -> tensor<384x32xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<384x384xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<384x32xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<384x384xf32>, tensor<384x32xf32>) -> tensor<384x32xf32>
    return %0 : tensor<384x32xf32>
}

//===----------------------------------------------------------------------===//
// O(N^2) matmul ops.
//===----------------------------------------------------------------------===//

func.func @dot_1x1024x1024() -> tensor<1x1024xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<1x1024xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<1024x1024xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<1x1024xf32>, tensor<1024x1024xf32>) -> tensor<1x1024xf32>
    return %0 : tensor<1x1024xf32>
}

func.func @dot_1x1024x2048() -> tensor<1x2048xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<1x1024xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<1024x2048xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<1x1024xf32>, tensor<1024x2048xf32>) -> tensor<1x2048xf32>
    return %0 : tensor<1x2048xf32>
}

func.func @dot_1x1024x3072() -> tensor<1x3072xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<1x1024xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<1024x3072xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<1x1024xf32>, tensor<1024x3072xf32>) -> tensor<1x3072xf32>
    return %0 : tensor<1x3072xf32>
}

func.func @dot_1x1024x512() -> tensor<1x512xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<1x1024xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<1024x512xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<1x1024xf32>, tensor<1024x512xf32>) -> tensor<1x512xf32>
    return %0 : tensor<1x512xf32>
}

func.func @dot_1x128x2() -> tensor<1x2xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<1x128xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<128x2xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<1x128xf32>, tensor<128x2xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
}

func.func @dot_1x256x512() -> tensor<1x512xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<1x256xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<256x512xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<1x256xf32>, tensor<256x512xf32>) -> tensor<1x512xf32>
    return %0 : tensor<1x512xf32>
}

func.func @dot_1x3072x1024() -> tensor<1x1024xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<1x3072xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<3072x1024xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<1x3072xf32>, tensor<3072x1024xf32>) -> tensor<1x1024xf32>
    return %0 : tensor<1x1024xf32>
}

func.func @dot_1x3072x512() -> tensor<1x512xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<1x3072xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<3072x512xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<1x3072xf32>, tensor<3072x512xf32>) -> tensor<1x512xf32>
    return %0 : tensor<1x512xf32>
}

func.func @dot_1x512x1024() -> tensor<1x1024xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<1x512xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<512x1024xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<1x512xf32>, tensor<512x1024xf32>) -> tensor<1x1024xf32>
    return %0 : tensor<1x1024xf32>
}

func.func @dot_1x512x3072() -> tensor<1x3072xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<1x512xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<512x3072xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<1x512xf32>, tensor<512x3072xf32>) -> tensor<1x3072xf32>
    return %0 : tensor<1x3072xf32>
}

func.func @dot_1x512x512() -> tensor<1x512xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<1x512xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<512x512xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<1x512xf32>, tensor<512x512xf32>) -> tensor<1x512xf32>
    return %0 : tensor<1x512xf32>
}

func.func @dot_1x528x128() -> tensor<1x128xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<1x528xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<528x128xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<1x528xf32>, tensor<528x128xf32>) -> tensor<1x128xf32>
    return %0 : tensor<1x128xf32>
}

func.func @dot_2x3072x512() -> tensor<2x512xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<2x3072xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<3072x512xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<2x3072xf32>, tensor<3072x512xf32>) -> tensor<2x512xf32>
    return %0 : tensor<2x512xf32>
}

func.func @dot_2x512x1024() -> tensor<2x1024xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<2x512xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<512x1024xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<2x512xf32>, tensor<512x1024xf32>) -> tensor<2x1024xf32>
    return %0 : tensor<2x1024xf32>
}

func.func @dot_2x512x3072() -> tensor<2x3072xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<2x512xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<512x3072xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<2x512xf32>, tensor<512x3072xf32>) -> tensor<2x3072xf32>
    return %0 : tensor<2x3072xf32>
}

func.func @dot_2x512x512() -> tensor<2x512xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<2x512xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<512x512xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<2x512xf32>, tensor<512x512xf32>) -> tensor<2x512xf32>
    return %0 : tensor<2x512xf32>
}

func.func @dot_2x528x512() -> tensor<2x512xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<2x528xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<528x512xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<2x528xf32>, tensor<528x512xf32>) -> tensor<2x512xf32>
    return %0 : tensor<2x512xf32>
}

func.func @dot_6x513x128() -> tensor<6x128xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<6x513xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<513x128xf32>
    %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<6x513xf32>, tensor<513x128xf32>) -> tensor<6x128xf32>
    return %0 : tensor<6x128xf32>
}
