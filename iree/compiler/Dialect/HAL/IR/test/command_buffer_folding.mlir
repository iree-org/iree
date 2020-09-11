// Tests folding and canonicalization of HAL command buffer ops.

// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @skip_command_buffer_device
func @skip_command_buffer_device() -> !hal.executable {
  %dev = hal.ex.shared_device : !hal.device
  %cmd = hal.command_buffer.create %dev, "OneShot", "Transfer|Dispatch" : !hal.command_buffer

  // CHECK-NOT: hal.command_buffer.device
  // CHECK: %[[EXECUTABLE:.+]] = hal.executable.lookup %dev, @executable_name : !hal.executable
  %0 = hal.command_buffer.device %cmd : !hal.device
  %exe = hal.executable.lookup %0, @executable_name : !hal.executable

  return %exe : !hal.executable
}
