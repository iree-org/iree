// RUN: iree-opt --iree-stablehlo-input-transformation-pipeline %s --verify-diagnostics

// A runtime amount is not folded, and dynamic_pad has no lowering yet.
func.func @dynamic_pad_runtime_low(%arg0: tensor<2x3x5xf32>, %arg1: tensor<f32>,
                                   %low: tensor<3xi64>) -> tensor<?x?x?xf32> {
  %high = stablehlo.constant dense<[0, 2, 1]> : tensor<3xi64>
  %interior = stablehlo.constant dense<0> : tensor<3xi64>
  // expected-error @+1 {{failed to legalize operation 'stablehlo.dynamic_pad' that was explicitly marked illegal}}
  %0 = "stablehlo.dynamic_pad"(%arg0, %arg1, %low, %high, %interior)
    : (tensor<2x3x5xf32>, tensor<f32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>)
    -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
