// RUN: iree-opt --split-input-file --iree-hal-verify-devices %s --mlir-print-local-scope --verify-diagnostics | FileCheck %s

// expected-error@+1 {{no HAL devices defined in the module}}
module @module {
  util.func private @func() -> ()
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
