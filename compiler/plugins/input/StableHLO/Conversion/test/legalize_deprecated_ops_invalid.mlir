// RUN: iree-opt --split-input-file --verify-diagnostics --iree-stablehlo-legalize-deprecated-ops %s
// RUN: iree-opt --split-input-file --verify-diagnostics --iree-stablehlo-input-transformation-pipeline %s

// The operation is verifier-valid, but its quantized element type cannot be
// used to build the scalar identity in the upstream deprecated-op replacement.
func.func @negative_quantized_unary(%input: tensor<2x!quant.uniform<i8:f32, 0.25>>) -> tensor<2x!quant.uniform<i8:f32, 0.25>> {
  // expected-error @+1 {{unsupported element type for unary einsum: '!quant.uniform<i8:f32, 2.500000e-01>'}}
  %r = "stablehlo.unary_einsum"(%input) {einsum_config = "i->i"} : (tensor<2x!quant.uniform<i8:f32, 0.25>>) -> tensor<2x!quant.uniform<i8:f32, 0.25>>
  return %r : tensor<2x!quant.uniform<i8:f32, 0.25>>
}
