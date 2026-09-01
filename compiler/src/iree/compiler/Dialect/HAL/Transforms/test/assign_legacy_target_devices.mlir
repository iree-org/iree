// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(iree-hal-assign-legacy-target-devices)' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-0
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(iree-hal-assign-legacy-target-devices{targetBackends=vmvx})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-1
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(iree-hal-assign-legacy-target-devices{targetBackends=vmvx,vmvx-inline})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-2
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(iree-hal-assign-legacy-target-devices{targetBackends=vmvx,vmvx})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-EQ
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(iree-hal-assign-legacy-target-devices{target-registry=xyz targetBackends=vmvx,vmvx})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-EQ
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(iree-hal-assign-legacy-target-devices{target-registry=global targetBackends=vmvx,vmvx})' %s | FileCheck %s --check-prefix=CHECK --check-prefix=TARGET-EQ

// CHECK: module
// TARGET-0: @module {
// TARGET-1: @module attributes {
// TARGET-1-SAME: hal.device.targets = [#hal.device.target<"local", [#hal.executable.target<"vmvx"
// TARGET-2: @module attributes {
// TARGET-2-SAME: hal.device.targets = [
// TARGET-2-SAME:   #hal.device.select<[
// TARGET-2-SAME:     #hal.device.target<"local", [#hal.executable.target<"vmvx"
// TARGET-2-SAME:     #hal.device.target<"local", [#hal.executable.target<"vmvx-inline"
// TARGET-EQ: @module attributes {
// TARGET-EQ-SAME: hal.device.targets = [#hal.device.target<"local",
module @module {}

// -----

// The pass is a no-op when targets are already specified.

// CHECK: module @module attributes {hal.device.targets = [#hal.device.target<"foo"
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
