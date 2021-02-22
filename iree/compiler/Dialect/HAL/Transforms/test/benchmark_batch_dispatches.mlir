// RUN: iree-opt -split-input-file -test-iree-hal-benchmark-batch-dispatches-2-times %s | IreeFileCheck %s

hal.variable @_executable : !hal.executable

// CHECK-LABEL: @multiple_reads_no_writes
//  CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer
func @multiple_reads_no_writes(%cmd : !hal.command_buffer) {
  // CHECK: %[[EXE:.+]] = hal.variable.load @_executable
  %exe = hal.variable.load @_executable : !hal.executable

  %c1 = constant 1 : index
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[0] workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[1] workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[2] workgroups([%c1, %c1, %c1])

  return
}

// CHECK: hal.command_buffer.dispatch<%[[CMD:.+]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[0] workgroups([%c1, %c1, %c1])
// CHECK: hal.command_buffer.dispatch<%[[CMD:.+]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[0] workgroups([%c1, %c1, %c1])
// CHECK: hal.command_buffer.dispatch<%[[CMD:.+]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[1] workgroups([%c1, %c1, %c1])
// CHECK: hal.command_buffer.dispatch<%[[CMD:.+]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[1] workgroups([%c1, %c1, %c1])
// CHECK: hal.command_buffer.dispatch<%[[CMD:.+]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[2] workgroups([%c1, %c1, %c1])
// CHECK: hal.command_buffer.dispatch<%[[CMD:.+]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[2] workgroups([%c1, %c1, %c1])
