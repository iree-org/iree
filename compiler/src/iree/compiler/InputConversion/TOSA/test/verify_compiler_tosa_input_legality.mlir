// RUN: iree-opt --split-input-file --iree-tosa-verify-compiler-input-legality --verify-diagnostics %s
// -verify-diagnostics

// expected-error@+1 {{one or more illegal operations were found in the compiler input}}
module {
func.func @illegal_tosa(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-note@+1 {{failed to legalize operation 'tosa.add' that was explicitly marked illegal}}
  %0 = tosa.add %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
}
