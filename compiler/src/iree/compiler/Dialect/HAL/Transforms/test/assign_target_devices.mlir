// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-hal-assign-target-devices)' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-0
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targets=vmvx})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-1
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targets=vmvx,vmvx-inline})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-2
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targets=vulkan,vulkan-spirv})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-EQ

// TARGET-1: #device_target_vmvx = #hal.device.target<"vmvx"

// TARGET-2: #device_target_vmvx = #hal.device.target<"vmvx"
// TARGET-2: #device_target_vmvx_inline = #hal.device.target<"vmvx-inline"

// TARGET-EQ: #device_target_vulkan = #hal.device.target<"vulkan"

// CHECK: module
// TARGET-0: @module {
// TARGET-1: @module attributes {
// TARGET-1-SAME: hal.device.targets = [#device_target_vmvx]
// TARGET-2: @module attributes {
// TARGET-2-SAME: hal.device.targets = [#device_target_vmvx, #device_target_vmvx_inline]}
// TARGET-EQ: @module attributes {
// TARGET-EQ-SAME: hal.device.targets = [#device_target_vulkan]}
module @module {}

// -----

// The pass is a no-op when targets are already specified.

// CHECK: #device_target_foo = #hal.device.target<"foo"
// CHECK: module @module attributes {hal.device.targets = [#device_target_foo]}
module @module attributes {
  hal.device.targets = [#hal.device.target<"foo">]
} {}
