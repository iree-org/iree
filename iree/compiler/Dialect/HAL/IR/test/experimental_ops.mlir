// RUN: iree-opt -allow-unregistered-dialect -split-input-file %s | iree-opt -allow-unregistered-dialect -split-input-file | FileCheck %s

// CHECK-LABEL: @shared_device
func @shared_device() -> !hal.device {
  // CHECK: %device = hal.ex.shared_device : !hal.device
  %device = hal.ex.shared_device : !hal.device
  return %device : !hal.device
}

// -----

// CHECK-LABEL: @submit_and_wait
func @submit_and_wait() {
  %0 = "test_hal.device"() : () -> !hal.device
  %1 = "test_hal.command_buffer"() : () -> !hal.command_buffer
  // CHECK: hal.ex.submit_and_wait %0, %1
  hal.ex.submit_and_wait %0, %1
  return
}
