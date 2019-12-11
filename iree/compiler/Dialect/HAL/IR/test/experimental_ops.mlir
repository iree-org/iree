// Tests printing and parsing of experimental ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @shared_device
func @shared_device() -> !ireex.ref<!hal.device> {
  // CHECK: %dev = hal.ex.shared_device : !ireex.ref<!hal.device>
  %dev = hal.ex.shared_device : !ireex.ref<!hal.device>
  return %dev : !ireex.ref<!hal.device>
}

// -----

// CHECK-LABEL: @cache_executable
hal.executable @foo {}
func @cache_executable() -> !ireex.ref<!hal.executable> {
  %0 = "test_hal.device"() : () -> !ireex.ref<!hal.device>
  // CHECK: %exe = hal.ex.cache_executable %0, @foo : !ireex.ref<!hal.executable>
  %exe = hal.ex.cache_executable %0, @foo : !ireex.ref<!hal.executable>
  return %exe : !ireex.ref<!hal.executable>
}

// -----

// CHECK-LABEL: @submit_and_wait
func @submit_and_wait() {
  %0 = "test_hal.device"() : () -> !ireex.ref<!hal.device>
  %1 = "test_hal.command_buffer"() : () -> !ireex.ref<!hal.command_buffer>
  // CHECK: hal.ex.submit_and_wait %0, %1
  hal.ex.submit_and_wait %0, %1
  return
}
