// RUN: iree-opt --verify-diagnostics --pass-pipeline="builtin.module(util.func(iree-global-opt-uninitialized-value-validation))" %s

util.func @matmul_128x128x128xf32(%input : tensor<128x128xf32>) -> (tensor<128x128xf32>) {
  %C = tensor.empty() : tensor<128x128xf32>
  // expected-error @+1 {{has an uninitialized operand (produced by a tensor.empty op)}}
  %0 = linalg.matmul ins(%input, %input : tensor<128x128xf32>, tensor<128x128xf32>) outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  util.return %0 : tensor<128x128xf32>
}
