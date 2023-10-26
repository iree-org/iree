// RUN: iree-opt --split-input-file -test-iree-hal-benchmark-batch-dispatches-2-times %s | FileCheck %s

util.global @_executable : !hal.executable

// CHECK-LABEL: @duplicate_dispatches
//  CHECK-SAME: (%[[CMD1:.+]]: !hal.command_buffer,
//  CHECK-SAME:  %[[CMD2:.+]]: !hal.command_buffer)
func.func @duplicate_dispatches(%cmd1 : !hal.command_buffer, %cmd2 : !hal.command_buffer) {
  // CHECK: %[[EXE:.+]] = util.global.load @_executable
  %exe = util.global.load @_executable : !hal.executable

  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  hal.command_buffer.dispatch<%cmd1 : !hal.command_buffer> target(%exe : !hal.executable)[0] workgroups([%c1, %c1, %c1])
  hal.command_buffer.execution_barrier<%cmd1 : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
  hal.command_buffer.dispatch<%cmd1 : !hal.command_buffer> target(%exe : !hal.executable)[1] workgroups([%c2, %c2, %c2])

  hal.command_buffer.dispatch<%cmd2 : !hal.command_buffer> target(%exe : !hal.executable)[2] workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch<%cmd2 : !hal.command_buffer> target(%exe : !hal.executable)[3] workgroups([%c2, %c2, %c2])
  hal.command_buffer.execution_barrier<%cmd2 : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")

  return
}

// CHECK: hal.command_buffer.dispatch<%[[CMD1]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[0] workgroups([%c1, %c1, %c1])
// CHECK: hal.command_buffer.execution_barrier<%[[CMD1]] : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
// CHECK: hal.command_buffer.dispatch<%[[CMD1]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[0] workgroups([%c1, %c1, %c1])
// CHECK: hal.command_buffer.execution_barrier<%[[CMD1]] : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")

// CHECK: hal.command_buffer.dispatch<%[[CMD1]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[1] workgroups([%c2, %c2, %c2])
// CHECK: hal.command_buffer.execution_barrier<%[[CMD1]] : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
// CHECK: hal.command_buffer.dispatch<%[[CMD1]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[1] workgroups([%c2, %c2, %c2])
// CHECK-NOT: hal.command_buffer.execution_barrier

// CHECK: hal.command_buffer.dispatch<%[[CMD2]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[2] workgroups([%c1, %c1, %c1])
// CHECK: hal.command_buffer.execution_barrier<%[[CMD2]] : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
// CHECK: hal.command_buffer.dispatch<%[[CMD2]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[2] workgroups([%c1, %c1, %c1])
// CHECK-NOT: hal.command_buffer.execution_barrier

// CHECK: hal.command_buffer.dispatch<%[[CMD2]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[3] workgroups([%c2, %c2, %c2])
// CHECK: hal.command_buffer.execution_barrier<%[[CMD2]] : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
// CHECK: hal.command_buffer.dispatch<%[[CMD2]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[3] workgroups([%c2, %c2, %c2])
// CHECK: hal.command_buffer.execution_barrier<%[[CMD2]] : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")

// -----

util.global @_executable : !hal.executable

// CHECK-LABEL: @nested_dispatch
//  CHECK-SAME: (%[[CMD1:.+]]: !hal.command_buffer,
//  CHECK-SAME:  %[[IDX:.+]]: index)
func.func @nested_dispatch(%cmd1 : !hal.command_buffer, %idx : index) {
  // CHECK: %[[EXE:.+]] = util.global.load @_executable
  %exe = util.global.load @_executable : !hal.executable

  %c1 = arith.constant 1 : index
  scf.index_switch %idx
  case 0 {
    hal.command_buffer.dispatch<%cmd1 : !hal.command_buffer> target(%exe : !hal.executable)[0] workgroups([%c1, %c1, %c1])
    scf.yield
  }
  default {
  }

  return
}

// CHECK: scf.index_switch
// CHECK: case 0 {
// CHECK:   hal.command_buffer.dispatch<%[[CMD1]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[0] workgroups([%c1, %c1, %c1])
// CHECK:   hal.command_buffer.execution_barrier<%[[CMD1]] : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
// CHECK:   hal.command_buffer.dispatch<%[[CMD1]] : !hal.command_buffer> target(%[[EXE]] : !hal.executable)[0] workgroups([%c1, %c1, %c1])
