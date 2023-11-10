// RUN: iree-opt --split-input-file --verify-diagnostics %s

util.global mutable @var : !hal.buffer
func.func @fn(%arg0: !hal.buffer_view) {
  // expected-error @+1 {{global "var" is '!hal.buffer' but store is '!hal.buffer_view'}}
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

// -----

hal.executable @ex_with_constants {
  hal.executable.variant @backend target(#hal.executable.target<"backend", "format">) {
    // expected-error @+1 {{must have one key for every result}}
    hal.executable.constant.block(%device: !hal.device) -> (i32, i32) as ("foo") {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      hal.return %c0, %c1 : i32, i32
    }
  }
}

// -----

hal.executable @ex_with_constants {
  hal.executable.variant @backend target(#hal.executable.target<"backend", "format">) {
    hal.executable.constant.block(%device: !hal.device) -> (i32, i32) as ("foo", "bar") {
      %c0 = arith.constant 0 : i32
      // expected-error @+1 {{return must have the same number of operands}}
      hal.return %c0 : i32
    }
  }
}

// -----

hal.executable @ex_with_constants {
  hal.executable.variant @backend target(#hal.executable.target<"backend", "format">) {
    hal.executable.constant.block(%device: !hal.device) -> i32 as "foo" {
      %c0 = arith.constant 0.0 : f32
      // expected-error @+1 {{parent expected result 0 to be 'i32' but returning 'f32'}}
      hal.return %c0 : f32
    }
  }
}

// -----

hal.executable @ex_with_constants {
  hal.executable.variant @backend target(#hal.executable.target<"backend", "format">) {
    // expected-error @+1 {{initializer must take a !hal.device or nothing}}
    hal.executable.constant.block(%device: !hal.device, %invalid: i32) -> i32 as "foo" {
      %c0 = arith.constant 0 : i32
      hal.return %c0 : i32
    }
  }
}

// -----

hal.executable @ex_with_constants {
  hal.executable.variant @backend target(#hal.executable.target<"backend", "format">) {
    // expected-error @+1 {{initializer must return only i32 values}}
    hal.executable.constant.block() -> f32 as "foo" {
      %c0 = arith.constant 0.0 : f32
      hal.return %c0 : f32
    }
  }
}
