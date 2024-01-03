// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @devices_count
func.func @devices_count() -> index {
  // CHECK: = hal.devices.count : index
  %device_count = hal.devices.count : index
  return %device_count : index
}

// -----

// CHECK-LABEL: @devices_get
// CHECK-SAME: (%[[INDEX:.+]]: index)
func.func @devices_get(%index: index) -> !hal.device {
  // CHECK: = hal.devices.get %[[INDEX]] : !hal.device
  %device = hal.devices.get %index : !hal.device
  return %device : !hal.device
}
