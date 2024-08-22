// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-hal-elide-redundant-commands))' %s | FileCheck %s

// Tests that redundant barriers are elided but barriers gaurding ops are not.

// CHECK-LABEL: @elideRedundantBarriers
// CHECK-SAME: (%[[CMD:.+]]: !hal.command_buffer, %[[BUFFER:.+]]: !hal.buffer)
util.func public @elideRedundantBarriers(%cmd: !hal.command_buffer, %buffer: !hal.buffer) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: hal.command_buffer.execution_barrier
  hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|Transfer|CommandRetire") target("CommandIssue|Dispatch|Transfer") flags("None")
  // CHECK-NOT: hal.command_buffer.execution_barrier
  hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|Transfer|CommandRetire") target("CommandIssue|Dispatch|Transfer") flags("None")
  // CHECK: hal.command_buffer.copy_buffer
  hal.command_buffer.copy_buffer<%cmd : !hal.command_buffer>
      source(%buffer : !hal.buffer)[%c0]
      target(%buffer : !hal.buffer)[%c0]
      length(%c1)
  // CHECK: hal.command_buffer.execution_barrier
  hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|Transfer|CommandRetire") target("CommandIssue|Dispatch|Transfer") flags("None")
  // CHECK: util.return
  util.return
}
