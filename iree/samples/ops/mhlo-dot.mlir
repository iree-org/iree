func @dot(%lhs: tensor<32x1024xf32>, %rhs: tensor<1024x64xf32>) -> tensor<32x64xf32> attributes { iree.module.export } {
  %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<32x1024xf32>, tensor<1024x64xf32>) -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}
