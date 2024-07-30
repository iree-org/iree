// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-hal-assign-legacy-target-devices)' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-0
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-hal-assign-legacy-target-devices{targetBackends=vmvx})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-1
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-hal-assign-legacy-target-devices{targetBackends=vmvx,vmvx-inline})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-2
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-hal-assign-legacy-target-devices{targetBackends=vmvx,vmvx})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-EQ

// TARGET-1: #device_target_local = #hal.device.target<"local"

// TARGET-2: #device_target_local = #hal.device.target<"local"
// TARGET-2: #device_target_vmvx_inline = #hal.device.target<"vmvx-inline"

// TARGET-EQ: #device_target_local = #hal.device.target<"local"

// CHECK: module
// TARGET-0: @module {
// TARGET-1: @module attributes {
// TARGET-1-SAME: hal.device.targets = [#device_target_local]
// TARGET-2: @module attributes {
// TARGET-2-SAME: hal.device.targets = [#hal.device.select<[#device_target_local, #device_target_vmvx_inline]> : !hal.device]
// TARGET-EQ: @module attributes {
// TARGET-EQ-SAME: hal.device.targets = [#device_target_local]}
module @module {}

// -----

// The pass is a no-op when targets are already specified.

// CHECK: #device_target_foo = #hal.device.target<"foo"
// CHECK: module @module attributes {hal.device.targets = [#device_target_foo]}
module @module attributes {
  hal.device.targets = [#hal.device.target<"foo">]
} {}

// -----

// The pass does nothing when one or more devices has already been defined.

// CHECK: module @module
// CHECK-NOT: hal.device.targets
module @module {
  // CHECK: @existing_device
  util.global private @existing_device : !hal.device
}
