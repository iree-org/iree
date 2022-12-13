// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @shared_device
func.func @shared_device() -> !hal.device {
  // CHECK: %device = hal.ex.shared_device : !hal.device
  %device = hal.ex.shared_device : !hal.device
  return %device : !hal.device
}
