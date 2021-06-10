func @abs(%input : tensor<f32>) -> (tensor<f32>) attributes { iree.module.export } {
  %result = absf %input : tensor<f32>
  return %result : tensor<f32>
}
