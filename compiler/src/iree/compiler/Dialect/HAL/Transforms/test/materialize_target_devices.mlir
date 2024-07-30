// RUN: iree-opt --split-input-file --iree-hal-materialize-target-devices %s --verify-diagnostics | FileCheck %s

// expected-error@+1 {{invalid device targets specified}}
module @module attributes {
  hal.device.targets = [
    "wrong_type"
  ]
} {
  util.func private @func() -> ()
}

// -----

// Modules without anything that needs an environment are OK as-is.

// CHECK: module @module
module @module {
  // CHECK-NEXT: hal.executable private @exe
  hal.executable private @exe {
    // CHECK-NEXT: hal.executable.variant public @embedded_elf_arm_64
    hal.executable.variant public @embedded_elf_arm_64 target(#hal.executable.target<"backend", "format", {}>) {}
  }
}

// -----

// Valid input with proper attributes for a single device.

// CHECK: #[[DEVICE_A:.+]] = #hal.device.target<"device_a"
#device_a = #hal.device.target<"device_a", [#hal.executable.target<"backend_a", "format_a">]>
// CHECK: #[[DEVICE_B:.+]] = #hal.device.target<"device_b"
#device_b = #hal.device.target<"device_b", [#hal.executable.target<"backend_b", "format_b">]>

// CHECK: module @module
// CHECK-NOT: hal.device.targets
// CHECK-SAME: stream.affinity.default = #hal.device.affinity<@__device_0>
module @module attributes {
  hal.device.targets = [
    #hal.device.select<[#device_a, #device_b]> : !hal.device
  ]
} {
  //      CHECK: util.global private @__device_0 = #hal.device.select<[
  // CHECK-SAME:   #[[DEVICE_A]],
  // CHECK-SAME:   #[[DEVICE_B]]
  // CHECK-SAME: ]> : !hal.device
}

// -----

// Multiple devices using device names.

// CHECK: #[[DEVICE_A:.+]] = #hal.device.target<"device_a"
#device_a = #hal.device.target<"device_a", [#hal.executable.target<"backend_a", "format_a">]>
// CHECK: #[[DEVICE_B:.+]] = #hal.device.target<"device_b"
#device_b = #hal.device.target<"device_b", [#hal.executable.target<"backend_b", "format_b">]>
// CHECK: #[[DEVICE_C:.+]] = #hal.device.target<"device_c"
#device_c = #hal.device.target<"device_c", [#hal.executable.target<"backend_c", "format_c">]>

// CHECK: module @module
// CHECK-NOT: hal.device.targets
// CHECK-SAME: stream.affinity.default = #hal.device.affinity<@device_a>
module @module attributes {
  hal.device.targets = {
    device_a = #device_a,
    device_bc = [#device_b, #device_c]
  }
} {
  // CHECK: util.global private @device_a = #[[DEVICE_A]]
  // CHECK: util.global private @device_bc = #hal.device.select<[#[[DEVICE_B]], #[[DEVICE_C]]]>
}

// -----

// Default device selection by name.

// CHECK: #[[DEVICE_A:.+]] = #hal.device.target<"device_a"
#device_a = #hal.device.target<"device_a", [#hal.executable.target<"backend_a", "format_a">]>
// CHECK: #[[DEVICE_B:.+]] = #hal.device.target<"device_b"
#device_b = #hal.device.target<"device_b", [#hal.executable.target<"backend_b", "format_b">]>

// CHECK: module @module
// CHECK-NOT: hal.device.targets
// CHECK-SAME: stream.affinity.default = #hal.device.affinity<@device_b>
module @module attributes {
  hal.device.targets = {
    device_a = #device_a,
    device_b = #device_b
  },
  hal.device.default = "device_b"
} {
  // CHECK: util.global private @device_a
  // CHECK: util.global private @device_b
}

// -----

// Default device selection by ordinal.

// CHECK: #[[DEVICE_A:.+]] = #hal.device.target<"device_a"
#device_a = #hal.device.target<"device_a", [#hal.executable.target<"backend_a", "format_a">]>
// CHECK: #[[DEVICE_B:.+]] = #hal.device.target<"device_b"
#device_b = #hal.device.target<"device_b", [#hal.executable.target<"backend_b", "format_b">]>

// CHECK: module @module
// CHECK-NOT: hal.device.targets
// CHECK-SAME: stream.affinity.default = #hal.device.affinity<@__device_1>
module @module attributes {
  hal.device.targets = [
    #device_a,
    #device_b
  ],
  hal.device.default = 1 : index
} {
  // CHECK: util.global private @__device_0
  // CHECK: util.global private @__device_1
}
