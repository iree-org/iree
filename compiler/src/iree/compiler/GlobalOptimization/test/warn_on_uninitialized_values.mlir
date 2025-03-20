// RUN: iree-opt --verify-diagnostics --pass-pipeline="builtin.module(util.func(iree-global-opt-warn-on-uninitialized-values))" %s

// Common error: forgetting to fill the input accumulator of a matmul.
util.func @matmul_uninitialized_accumulator(%input : tensor<128x128xf32>) -> (tensor<128x128xf32>) {
  %C = tensor.empty() : tensor<128x128xf32>
  // expected-warning @+1 {{reads uninitialized values from an operand produced by a tensor.empty op}}
  %0 = linalg.matmul ins(%input, %input : tensor<128x128xf32>, tensor<128x128xf32>) outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  util.return %0 : tensor<128x128xf32>
}

// This testcase differs from the previous one in that the empty accumulator
// gets transposed. This tests that the linalg.transpose tolerates its empty
// input (because the only op in the body that reads the uninitialized value is
// the terminator). This test then also observes the current false negative
// as the matmul is consuming uninitialized accumulator values and that is not
// currently diagnosed.
util.func @tolerate_transpse_of_uninitialized_and_then_false_negative(%input : tensor<128x128xf32>) -> (tensor<128x128xf32>) {
  %C = tensor.empty() : tensor<128x128xf32>
  %D = tensor.empty() : tensor<128x128xf32>
  %T = linalg.transpose ins(%C : tensor<128x128xf32>) outs(%D : tensor<128x128xf32>) permutation = [1, 0]
  %0 = linalg.matmul ins(%input, %input : tensor<128x128xf32>, tensor<128x128xf32>) outs(%T : tensor<128x128xf32>) -> tensor<128x128xf32>
  util.return %0 : tensor<128x128xf32>
}
