// RUN: iree-opt --split-input-file --iree-stream-conversion --verify-diagnostics %s

util.func public @unpacked_buffer_view_import(%view: !hal.buffer_view) -> tensor<4xi4, #iree_encoding.unpacked_storage> {
  // expected-error @+1 {{failed to legalize operation 'hal.tensor.import'}}
  %result = hal.tensor.import %view : !hal.buffer_view -> tensor<4xi4, #iree_encoding.unpacked_storage>
  util.return %result : tensor<4xi4, #iree_encoding.unpacked_storage>
}

// -----

util.func public @unpacked_buffer_view_export(%tensor: tensor<4xi4, #iree_encoding.unpacked_storage>) -> !hal.buffer_view {
  // expected-error @+1 {{failed to legalize operation 'hal.tensor.export'}}
  %result = hal.tensor.export %tensor : tensor<4xi4, #iree_encoding.unpacked_storage> -> !hal.buffer_view
  util.return %result : !hal.buffer_view
}
