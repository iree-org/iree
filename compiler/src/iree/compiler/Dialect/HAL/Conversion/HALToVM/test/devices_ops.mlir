// RUN: iree-opt --split-input-file --iree-convert-hal-to-vm --canonicalize --iree-vm-target-index-bits=32 %s | FileCheck %s

// CHECK-LABEL: @devices_count
func.func @devices_count() -> index {
  // CHECK: = vm.call @hal.devices.count() {nosideeffects} : () -> i32
  %device_count = hal.devices.count : index
  return %device_count : index
}

// -----

// CHECK-LABEL: @devices_get
// CHECK-SAME: (%[[INDEX:.+]]: i32)
func.func @devices_get(%index: index) -> !hal.device {
  // CHECK: = vm.call @hal.devices.get(%[[INDEX]]) {nosideeffects} : (i32) -> !vm.ref<!hal.device>
  %device = hal.devices.get %index : !hal.device
  return %device : !hal.device
}
