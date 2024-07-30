// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(iree-hal-assign-target-devices)' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-0
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=device})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-1
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=device_a,device_b})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-2
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=device_a[0],device_a[1]})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-ORDINALS
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=#hal.device.target<"local">})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-ATTR
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=#hal.device.alias<"device_a">})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-ALIAS
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices={"device_a,#hal.device.alias<"device_b">"}})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-SELECT
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=device_a=#hal.device.alias<"device_a">,"device_bc=device_b,#hal.device.alias<"device_c">"})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-SELECT-MULTI

// CHECK: module
// TARGET-0-NOT: hal.device.targets
// TARGET-1: hal.device.targets = [#hal.device.alias<"device"> : !hal.device]
// TARGET-2: hal.device.targets = [#hal.device.alias<"device_a"> : !hal.device, #hal.device.alias<"device_b"> : !hal.device]}
// TARGET-ORDINALS: hal.device.targets = [#hal.device.alias<"device_a"[0]> : !hal.device, #hal.device.alias<"device_a"[1]> : !hal.device]}
// TARGET-ATTR: hal.device.targets = [#hal.device.target<"local"> : !hal.device]
// TARGET-ALIAS: hal.device.targets = [#hal.device.alias<"device_a"> : !hal.device]
// TARGET-SELECT: hal.device.targets = [#hal.device.select<[#hal.device.alias<"device_a"> : !hal.device, #hal.device.alias<"device_b"> : !hal.device]> : !hal.device]
// TARGET-SELECT-MULTI: hal.device.targets = {
// TARGET-SELECT-MULTI-SAME: device_a = #hal.device.alias<"device_a"> : !hal.device,
// TARGET-SELECT-MULTI-SAME: device_bc = #hal.device.select<[#hal.device.alias<"device_b"> : !hal.device, #hal.device.alias<"device_c"> : !hal.device]> : !hal.device
// TARGET-SELECT-MULTI-SAME: }
module @module {
  util.global private @tensor_global : tensor<4xf32>
}

// -----

// The pass is a no-op when targets are already specified.

// CHECK: module @module attributes {
// CHECK-SAME: hal.device.targets = [#hal.device.target<"foo"> : !hal.device]
module @module attributes {
  hal.device.targets = [#hal.device.target<"foo"> : !hal.device]
} {}

// -----

// The pass does nothing when one or more devices has already been defined.

// CHECK: module @module
// CHECK-NOT: hal.device.targets
module @module {
  // CHECK: @existing_device
  util.global private @existing_device : !hal.device
}
