// Tests printing and parsing of experimental ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @shared_device
func @shared_device() -> !iree.ref<!hal.device> {
  // CHECK: %dev = hal.ex.shared_device : !iree.ref<!hal.device>
  %dev = hal.ex.shared_device : !iree.ref<!hal.device>
  return %dev : !iree.ref<!hal.device>
}

// -----

// CHECK-LABEL: @cache_executable
hal.executable @foo {}
func @cache_executable() -> !iree.ref<!hal.executable> {
  %0 = "test_hal.device"() : () -> !iree.ref<!hal.device>
  // CHECK: %exe = hal.ex.cache_executable %0, @foo : !iree.ref<!hal.executable>
  %exe = hal.ex.cache_executable %0, @foo : !iree.ref<!hal.executable>
  return %exe : !iree.ref<!hal.executable>
}

// -----

// CHECK-LABEL: @submit_and_wait
func @submit_and_wait() {
  %0 = "test_hal.device"() : () -> !iree.ref<!hal.device>
  %1 = "test_hal.command_buffer"() : () -> !iree.ref<!hal.command_buffer>
  // CHECK: hal.ex.submit_and_wait %0, %1
  hal.ex.submit_and_wait %0, %1
  return
}
