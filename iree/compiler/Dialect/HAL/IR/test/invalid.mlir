// RUN: iree-opt -split-input-file -verify-diagnostics %s

hal.variable @var mutable : !hal.buffer
func @fn(%arg0: !hal.buffer_view) {
  // expected-error @+1 {{variable var is '!hal.buffer' but store is '!hal.buffer_view'}}
  hal.variable.store %arg0, @var : !hal.buffer_view
  return
}

// -----
hal.variable @var mutable : !hal.buffer
func @fn(%arg0: !hal.buffer_view) {
  %0 = hal.variable.address @var_indirect_with_buffer_view_store : !util.ptr<!hal.buffer>
  // expected-error @+1 {{variable pointer is '!hal.buffer' but store is '!hal.buffer_view'}}
  hal.variable.store.indirect %arg0, %0 : !hal.buffer_view -> !util.ptr<!hal.buffer>
  return
}
