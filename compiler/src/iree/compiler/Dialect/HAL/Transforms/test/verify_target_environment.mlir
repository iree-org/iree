// RUN: iree-opt --split-input-file --iree-hal-verify-target-environment %s --verify-diagnostics | FileCheck %s

// expected-error@+1 {{no HAL target devices specified}}
module @module {
  util.func private @func() -> ()
}

// -----

// expected-error@+1 {{no HAL target devices specified}}
module @module attributes {hal.device.targets = []} {
  util.func private @func() -> ()
}

// -----

// expected-error@+1 {{invalid target attr type}}
module @module attributes {hal.device.targets = ["wrong_type"]} {
  util.func private @func() -> ()
}

// -----

// expected-error@+1 {{unregistered target device "foo"}}
module @module attributes {hal.device.targets = [#hal.device.target<"foo">]} {
  util.func private @func() -> ()
}

// -----

// Valid input with proper attributes.

// CHECK: #device_target_vmvx = #hal.device.target<"vmvx">
#device_target_vmvx = #hal.device.target<"vmvx">

// CHECK: module @module attributes {hal.device.targets = [#device_target_vmvx]}
module @module attributes {hal.device.targets = [#device_target_vmvx]} {
  util.func private @func() -> ()
}

// -----

// Modules without anything that needs an environment are OK.

#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {}>

// CHECK: module @module
module @module {
  // CHECK-NEXT: hal.executable private @exe
  hal.executable private @exe {
    // CHECK-NEXT: hal.executable.variant public @embedded_elf_arm_64
    hal.executable.variant public @embedded_elf_arm_64 target(#executable_target) {}
  }
}
