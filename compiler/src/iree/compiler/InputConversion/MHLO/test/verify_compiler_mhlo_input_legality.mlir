// RUN: iree-opt --split-input-file --iree-mhlo-verify-compiler-input-legality --verify-diagnostics %s
// -verify-diagnostics

// expected-error@+1 {{one or more illegal operations were found in the compiler input}}
module {
func.func @illegal_chlo(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-note@+1 {{failed to legalize operation 'chlo.broadcast_add' that was explicitly marked illegal}}
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
}

// -----
// expected-error@+1 {{one or more illegal operations were found in the compiler input}}
module {
func.func @illegal_mhlo(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-note@+1 {{failed to legalize operation 'mhlo.add' that was explicitly marked illegal}}
  %0 = mhlo.add %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
}
