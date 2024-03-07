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

// Valid input with proper attributes.

// CHECK: #device_target_llvm_cpu = #hal.device.target<"llvm-cpu">
#device_target_llvm_cpu = #hal.device.target<"llvm-cpu">
// CHECK: #device_target_vmvx = #hal.device.target<"vmvx">
#device_target_vmvx = #hal.device.target<"vmvx">

// CHECK: module @module
// CHECK-NOT: hal.device.targets
module @module attributes {
  hal.device.targets = [
    #device_target_llvm_cpu,
    #device_target_vmvx
  ]
} {
  //      CHECK: util.global private @__device.0 = #hal.device.select<[
  // CHECK-SAME:   #device_target_llvm_cpu,
  // CHECK-SAME:   #device_target_vmvx
  // CHECK-SAME: ]> : !hal.device

  // CHECK: util.global private @tensor_global
  // CHECK-SAME: stream.affinity = #hal.device.affinity<@__device_0>
  util.global private @tensor_global : tensor<4xf32>

  // CHECK: util.global private @primitive_global
  // CHECK-NOT: stream.affinity
  util.global private @primitive_global : i32

  // CHECK: util.func private @func
  // CHECK-SAME: stream.affinity = #hal.device.affinity<@__device_0>
  util.func private @func() -> ()
}

// -----

// Modules without anything that needs an environment are OK.

// CHECK: module @module
module @module {
  // CHECK-NEXT: hal.executable private @exe
  hal.executable private @exe {
    // CHECK-NEXT: hal.executable.variant public @embedded_elf_arm_64
    hal.executable.variant public @embedded_elf_arm_64 target(#hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {}>) {}
  }
}
