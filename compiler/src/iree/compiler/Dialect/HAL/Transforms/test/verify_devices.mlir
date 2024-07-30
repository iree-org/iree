// RUN: iree-opt --split-input-file --iree-hal-verify-devices %s --mlir-print-local-scope --verify-diagnostics | FileCheck %s

// Tests that modules without tensors don't need devices.

module @module {
  // CHECK: util.func private @func
  util.func private @func() -> ()
}

// -----

// TODO(multi-device): find a way to verify that devices exist if they need to.
// Currently the check is disabled as it's difficult to tell if a device will be
// needed by the time we get to the HAL layer: plugins may absorb things, etc.
// NO-expected-errorx@+1 {{no HAL devices defined in the module}}
module @module {
  util.func private @func() -> () {
    arith.constant dense<1.0> : tensor<4xf32>
    util.return
  }
}

// -----

module @module {
  // expected-error@+1 {{unregistered target device "__unregistered__"}}
  util.global private @device = #hal.device.target<"__unregistered__"> : !hal.device
  util.func private @func() -> () attributes {
    stream.affinity = #hal.device.affinity<@device>
  }
}

// -----

module @module {
  // expected-error@+1 {{unregistered target device "__unregistered__"}}
  util.global private @device = #hal.device.select<[
    #hal.device.target<"vmvx"> : !hal.device,
    #hal.device.target<"__unregistered__"> : !hal.device
  ]> : !hal.device
  util.func private @func() -> () attributes {
    stream.affinity = #hal.device.affinity<@device>
  }
}

// -----

// Valid input with proper attributes.

// CHECK: module @module
module @module {
  util.global private @device = #hal.device.target<"vmvx"> : !hal.device
  util.global private @optional = #hal.device.fallback<@device> : !hal.device
  util.global private @ordinal = #hal.device.ordinal<0> : !hal.device
  util.global private @selected = #hal.device.select<[
    #hal.device.target<"llvm-cpu"> : !hal.device,
    #hal.device.target<"vmvx"> : !hal.device
  ]> : !hal.device
  util.func private @func() -> () attributes {
    stream.affinity = #hal.device.affinity<@device>
  }
}

// -----

// Modules without anything that needs an environment are OK.

// CHECK: module @module
module @module {
  hal.executable private @exe {
    hal.executable.variant public @embedded_elf_arm_64 target(#hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {}>) {}
  }
}
