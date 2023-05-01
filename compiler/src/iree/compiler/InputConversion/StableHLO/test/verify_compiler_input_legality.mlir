// RUN: iree-opt --split-input-file --iree-stablehlo-verify-compiler-input-legality \
// RUN:   --verify-diagnostics %s

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
func.func @illegal_stablehlo(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-note@+1 {{failed to legalize operation 'stablehlo.add' that was explicitly marked illegal}}
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
}

// -----
// expected-error@+1 {{one or more illegal operations were found in the compiler input}}
module {
func.func @illegal_shape(%arg0: tensor<*xf32>) -> index {
  // expected-note@+1 {{failed to legalize operation 'shape.shape_of' that was explicitly marked illegal}}
  %arg_shape = shape.shape_of %arg0 : tensor<*xf32> -> tensor<?xindex>
  %rank = shape.rank %arg_shape : tensor<?xindex> -> index
  return %rank : index
}
}
