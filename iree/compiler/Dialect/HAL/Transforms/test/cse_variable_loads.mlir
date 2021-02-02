// RUN: iree-opt -split-input-file -iree-hal-cse-variable-loads %s | IreeFileCheck %s

// CHECK: hal.variable @_executable_0 : !hal.executable
hal.variable @_executable_0 : !hal.executable
// CHECK-LABEL: @multiple_reads_no_writes
func @multiple_reads_no_writes() {
  // CHECK-NEXT: %0 = hal.variable.load @_executable_0 : !hal.executable
  %0 = hal.variable.load @_executable_0 : !hal.executable
  // CHECK-NOT: hal.variable.load
  %1 = hal.variable.load @_executable_0 : !hal.executable
  // CHECK-NOT: hal.variable.load
  %2 = hal.variable.load @_executable_0 : !hal.executable

  %c1 = constant 1 : index
  %dev = hal.ex.shared_device : !hal.device
  %cmd = hal.command_buffer.create %dev, "OneShot", "Transfer|Dispatch" : !hal.command_buffer
  hal.command_buffer.begin %cmd
  // CHECK: hal.command_buffer.dispatch %cmd, %0, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  hal.command_buffer.dispatch %cmd, %0, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  // CHECK-NEXT: hal.command_buffer.dispatch %cmd, %0, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  hal.command_buffer.dispatch %cmd, %1, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  // CHECK-NEXT: hal.command_buffer.dispatch %cmd, %0, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  hal.command_buffer.dispatch %cmd, %2, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  hal.command_buffer.end %cmd
  return
}

// -----

// CHECK: hal.variable @_executable_0 : !hal.executable
hal.variable @_executable_0 : !hal.executable
// CHECK: hal.variable @_executable_1 : !hal.executable
hal.variable @_executable_1 : !hal.executable
// CHECK: hal.variable @_executable_2 : !hal.executable
hal.variable @_executable_2 : !hal.executable
// CHECK-LABEL: @different_variables_are_not_eliminated
func @different_variables_are_not_eliminated() {
  // CHECK-NEXT: %0 = hal.variable.load @_executable_0 : !hal.executable
  %0 = hal.variable.load @_executable_0 : !hal.executable
  // CHECK-NEXT: %1 = hal.variable.load @_executable_1 : !hal.executable
  %1 = hal.variable.load @_executable_1 : !hal.executable
  // CHECK-NEXT: %2 = hal.variable.load @_executable_2 : !hal.executable
  %2 = hal.variable.load @_executable_2 : !hal.executable

  %c1 = constant 1 : index
  %dev = hal.ex.shared_device : !hal.device
  %cmd = hal.command_buffer.create %dev, "OneShot", "Transfer|Dispatch" : !hal.command_buffer
  hal.command_buffer.begin %cmd
  // CHECK: hal.command_buffer.dispatch %cmd, %0, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  hal.command_buffer.dispatch %cmd, %0, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  // CHECK-NEXT: hal.command_buffer.dispatch %cmd, %1, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  hal.command_buffer.dispatch %cmd, %1, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  // CHECK-NEXT: hal.command_buffer.dispatch %cmd, %2, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  hal.command_buffer.dispatch %cmd, %2, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  hal.command_buffer.end %cmd
  return
}

// -----

hal.executable @exe {}

// CHECK: hal.variable @_executable_0 mutable : !hal.executable
hal.variable @_executable_0 mutable : !hal.executable
// CHECK-LABEL: @writes_prevent_cse
func @writes_prevent_cse() {
  // CHECK-NEXT: %0 = hal.variable.load @_executable_0 : !hal.executable
  %0 = hal.variable.load @_executable_0 : !hal.executable
  // CHECK-NEXT: %1 = hal.variable.load @_executable_0 : !hal.executable
  %1 = hal.variable.load @_executable_0 : !hal.executable

  %dev = hal.ex.shared_device : !hal.device
  %exe = hal.executable.lookup %dev, @exe : !hal.executable
  // CHECK: hal.variable.store %exe, @_executable_0 : !hal.executable
  hal.variable.store %exe, @_executable_0 : !hal.executable

  %c1 = constant 1 : index
  %cmd = hal.command_buffer.create %dev, "OneShot", "Transfer|Dispatch" : !hal.command_buffer
  hal.command_buffer.begin %cmd
  // CHECK: hal.command_buffer.dispatch %cmd, %0, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  hal.command_buffer.dispatch %cmd, %0, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  // CHECK-NEXT: hal.command_buffer.dispatch %cmd, %1, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  hal.command_buffer.dispatch %cmd, %1, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  hal.command_buffer.end %cmd
  return
}

// -----

// CHECK: hal.variable @_executable_0 : !hal.executable
hal.variable @_executable_0 : !hal.executable
// CHECK-LABEL: @reads_in_blocks
func @reads_in_blocks() {
  // load should be hoisted to the entry block
  // CHECK-NEXT: %0 = hal.variable.load @_executable_0 : !hal.executable

  %c1 = constant 1 : index
  %dev = hal.ex.shared_device : !hal.device
  %cmd = hal.command_buffer.create %dev, "OneShot", "Transfer|Dispatch" : !hal.command_buffer
  hal.command_buffer.begin %cmd

  %i1 = constant 1 : i1
  cond_br %i1, ^bb1, ^bb2
^bb1:
  // CHECK-NOT: hal.variable.load
  %0 = hal.variable.load @_executable_0 : !hal.executable
  // CHECK: hal.command_buffer.dispatch %cmd, %0, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  hal.command_buffer.dispatch %cmd, %0, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  br ^bb3
^bb2:
  // CHECK-NOT: hal.variable.load
  %1 = hal.variable.load @_executable_0 : !hal.executable
  // CHECK: hal.command_buffer.dispatch %cmd, %0, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  hal.command_buffer.dispatch %cmd, %1, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  br ^bb3
^bb3:
  hal.command_buffer.end %cmd
  return
}
