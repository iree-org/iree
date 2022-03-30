// RUN: iree-opt -split-input-file -verify-diagnostics %s

util.global mutable @var : !hal.buffer
func.func @fn(%arg0: !hal.buffer_view) {
  // expected-error @+1 {{global var is '!hal.buffer' but store is '!hal.buffer_view'}}
  util.global.store %arg0, @var : !hal.buffer_view
  return
}

// -----
util.global mutable @var : !hal.buffer
func.func @fn(%arg0: !hal.buffer_view) {
  %0 = util.global.address @var : !util.ptr<!hal.buffer>
  // expected-error @+1 {{global pointer is '!hal.buffer' but store is '!hal.buffer_view'}}
  util.global.store.indirect %arg0, %0 : !hal.buffer_view -> !util.ptr<!hal.buffer>
  return
}
