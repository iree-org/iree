// RUN: iree-opt --split-input-file --verify-diagnostics %s

util.func public @barrier_start_shape_mismatch(%arg0: tensor<4x8xf32>) -> tensor<8x4xf32> {
  // expected-error@+1 {{'iree_tensor_ext.compute_barrier.start' op value and result types must match}}
  %0 = iree_tensor_ext.compute_barrier.start %arg0 : tensor<4x8xf32> -> tensor<8x4xf32>
  util.return %0 : tensor<8x4xf32>
}

// -----

util.func public @barrier_start_missing_dynamic_dims(%arg0: tensor<?x?xf32>, %dim0: index) -> tensor<?x?xf32> {
  // expected-error@+1 {{'iree_tensor_ext.compute_barrier.start' op value set has 2 dynamic dimensions but only 1 dimension values are attached}}
  %0 = iree_tensor_ext.compute_barrier.start %arg0 : tensor<?x?xf32>{%dim0} -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}

// -----

util.func public @barrier_start_static_with_dims(%arg0: tensor<4x8xf32>, %dim0: index) -> tensor<4x8xf32> {
  // expected-error@+1 {{'iree_tensor_ext.compute_barrier.start' op value set has 0 dynamic dimensions but only 1 dimension values are attached}}
  %0 = iree_tensor_ext.compute_barrier.start %arg0 : tensor<4x8xf32>{%dim0} -> tensor<4x8xf32>
  util.return %0 : tensor<4x8xf32>
}

// -----

util.func public @barrier_end_shape_mismatch(%arg0: tensor<4x8xf32>) -> tensor<8x4xf32> {
  // expected-error@+1 {{'iree_tensor_ext.compute_barrier.end' op value and result types must match}}
  %0 = iree_tensor_ext.compute_barrier.end %arg0 : tensor<4x8xf32> -> tensor<8x4xf32>
  util.return %0 : tensor<8x4xf32>
}

// -----

util.func public @barrier_end_missing_dynamic_dims(%arg0: tensor<?x?xf32>, %dim0: index) -> tensor<?x?xf32> {
  // expected-error@+1 {{'iree_tensor_ext.compute_barrier.end' op value set has 2 dynamic dimensions but only 1 dimension values are attached}}
  %0 = iree_tensor_ext.compute_barrier.end %arg0 : tensor<?x?xf32>{%dim0} -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}

// -----

util.func public @barrier_end_static_with_dims(%arg0: tensor<4x8xf32>, %dim0: index) -> tensor<4x8xf32> {
  // expected-error@+1 {{'iree_tensor_ext.compute_barrier.end' op value set has 0 dynamic dimensions but only 1 dimension values are attached}}
  %0 = iree_tensor_ext.compute_barrier.end %arg0 : tensor<4x8xf32>{%dim0} -> tensor<4x8xf32>
  util.return %0 : tensor<4x8xf32>
}
