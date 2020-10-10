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

// -----

// CHECK-LABEL: @fold_buffer_subspan_into_push_descriptor_set
// CHECK-SAME: [[BASE_BUFFER:%[a-z0-9]+]]: !hal.buffer
func @fold_buffer_subspan_into_push_descriptor_set(
    %cmd : !hal.command_buffer,
    %layout : !hal.executable_layout,
    %buffer : !hal.buffer
  ) {
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c4096 = constant 4096 : index
  %c8000 = constant 8000 : index
  %c262140 = constant 262140 : index
  %c262144 = constant 262144 : index
  %subspan = hal.buffer.subspan %buffer, %c4096, %c262144 : !hal.buffer
  // CHECK: hal.command_buffer.push_descriptor_set {{.+}}, bindings=[
  hal.command_buffer.push_descriptor_set %cmd, %layout, set=0, bindings=[
    // 0 + 4096:
    // CHECK-SAME: 0 = ([[BASE_BUFFER]], %c4096, %c8000)
    0 = (%subspan, %c0, %c8000),
    // 4096 + 4:
    // CHECK-SAME: 1 = ([[BASE_BUFFER]], %c4100, %c262140)
    1 = (%subspan, %c4, %c262140),
    // No change:
    // CHECK-SAME: 2 = ([[BASE_BUFFER]], %c4096, %c262144)
    2 = (%buffer, %c4096, %c262144)
  ]
  return
}
