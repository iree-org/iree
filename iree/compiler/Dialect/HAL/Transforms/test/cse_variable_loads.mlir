// RUN: iree-opt -split-input-file -iree-hal-cse-variable-loads %s | IreeFileCheck %s

// CHECK: hal.variable @_executable : !hal.executable
hal.variable @_executable : !hal.executable

// CHECK-LABEL: @multiple_reads_no_writes
// CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer
func @multiple_reads_no_writes(%cmd : !hal.command_buffer) {
  // CHECK-NEXT: %[[EXE:.+]] = hal.variable.load @_executable : !hal.executable
  %exe0 = hal.variable.load @_executable : !hal.executable
  // CHECK-NOT: hal.variable.load
  %exe1 = hal.variable.load @_executable : !hal.executable
  // CHECK-NOT: hal.variable.load
  %exe2 = hal.variable.load @_executable : !hal.executable

  %c1 = constant 1 : index
  // CHECK: hal.command_buffer.dispatch<%[[CMD]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[0]
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe0 : !hal.executable)[0] workgroups([%c1, %c1, %c1])
  // CHECK: hal.command_buffer.dispatch<%[[CMD]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[1]
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe1 : !hal.executable)[1] workgroups([%c1, %c1, %c1])
  // CHECK: hal.command_buffer.dispatch<%[[CMD]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[2]
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe2 : !hal.executable)[2] workgroups([%c1, %c1, %c1])

  return
}

// -----

// CHECK: hal.variable @_executable_0 : !hal.executable
hal.variable @_executable_0 : !hal.executable
// CHECK: hal.variable @_executable_1 : !hal.executable
hal.variable @_executable_1 : !hal.executable

// CHECK-LABEL: @different_variables_are_not_eliminated
// CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer
func @different_variables_are_not_eliminated(%cmd : !hal.command_buffer) {
  // CHECK-NEXT: %[[EXE0:.+]] = hal.variable.load @_executable_0 : !hal.executable
  %exe0 = hal.variable.load @_executable_0 : !hal.executable
  // CHECK-NEXT: %[[EXE1:.+]] = hal.variable.load @_executable_1 : !hal.executable
  %exe1 = hal.variable.load @_executable_1 : !hal.executable

  %c1 = constant 1 : index
  // CHECK: hal.command_buffer.dispatch<%[[CMD]] : !hal.command_buffer> target(%[[EXE0]] : !hal.executable)[0]
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe0 : !hal.executable)[0] workgroups([%c1, %c1, %c1])
  // CHECK: hal.command_buffer.dispatch<%[[CMD]] : !hal.command_buffer> target(%[[EXE1]] : !hal.executable)[1]
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe1 : !hal.executable)[1] workgroups([%c1, %c1, %c1])

  return
}

// -----

hal.executable @exe {}

// CHECK: hal.variable @_executable mutable : !hal.executable
hal.variable @_executable mutable : !hal.executable

// CHECK-LABEL: @writes_prevent_cse
// CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer
func @writes_prevent_cse(%cmd : !hal.command_buffer) {
  // CHECK-NEXT: %[[EXE0:.+]] = hal.variable.load @_executable : !hal.executable
  %exe0 = hal.variable.load @_executable : !hal.executable
  // CHECK-NEXT: %[[EXE1:.+]] = hal.variable.load @_executable : !hal.executable
  %exe1 = hal.variable.load @_executable : !hal.executable

  %dev = hal.ex.shared_device : !hal.device
  %exe = hal.executable.lookup device(%dev : !hal.device) executable(@exe) : !hal.executable
  // CHECK: hal.variable.store %exe, @_executable : !hal.executable
  hal.variable.store %exe, @_executable : !hal.executable

  %c1 = constant 1 : index
  // CHECK: hal.command_buffer.dispatch<%[[CMD]] : !hal.command_buffer> target(%[[EXE0]] : !hal.executable)[0]
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe0 : !hal.executable)[0] workgroups([%c1, %c1, %c1])
  // CHECK: hal.command_buffer.dispatch<%[[CMD]] : !hal.command_buffer> target(%[[EXE1]] : !hal.executable)[1]
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe1 : !hal.executable)[1] workgroups([%c1, %c1, %c1])

  return
}

// -----

// CHECK: hal.variable @_executable : !hal.executable
hal.variable @_executable : !hal.executable

// CHECK-LABEL: @reads_in_blocks
// CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer
func @reads_in_blocks(%cmd : !hal.command_buffer) {
  // load should be hoisted to the entry block
  // CHECK-NEXT: %[[EXE:.+]] = hal.variable.load @_executable : !hal.executable

  %c1 = constant 1 : index
  %i1 = constant 1 : i1
  cond_br %i1, ^bb1, ^bb2
^bb1:
  // CHECK-NOT: hal.variable.load
  %exe0 = hal.variable.load @_executable : !hal.executable
  // CHECK: hal.command_buffer.dispatch<%[[CMD]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[0]
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe0 : !hal.executable)[0] workgroups([%c1, %c1, %c1])
  br ^bb3
^bb2:
  // CHECK-NOT: hal.variable.load
  %exe1 = hal.variable.load @_executable : !hal.executable
  // CHECK: hal.command_buffer.dispatch<%[[CMD]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[1]
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe1 : !hal.executable)[1] workgroups([%c1, %c1, %c1])
  br ^bb3
^bb3:
  return
}
