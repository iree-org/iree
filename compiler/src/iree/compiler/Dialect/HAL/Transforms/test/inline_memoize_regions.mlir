// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-hal-inline-memoize-regions,canonicalize))' %s | FileCheck %s

// CHECK-LABEL: @constant_memoize
util.func public @constant_memoize(%device: !hal.device, %affinity: i64) -> index {
  // CHECK-NOT: hal.device.memoize
  %result = hal.device.memoize<%device : !hal.device> affinity(%affinity) -> index {
    // CHECK: %[[C4:.+]] = arith.constant 4
    %c4 = arith.constant 4 : index
    // CHECK-NOT: hal.return
    hal.return %c4 : index
  }
  // CHECK: util.return %[[C4]]
  util.return %result : index
}

// -----

// CHECK-LABEL: @command_buffer_memoize
util.func public @command_buffer_memoize(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64)
    %device: !hal.device, %affinity: i64) -> !hal.command_buffer {
  // CHECK-NOT: hal.device.memoize
  %result = hal.device.memoize<%device : !hal.device> affinity(%affinity) -> !hal.command_buffer {
    // CHECK: %[[CMD:.+]] = hal.command_buffer.create
    // CHECK-SAME: device(%[[DEVICE]] : !hal.device)
    // CHECK-SAME: mode(OneShot)
    // CHECK-SAME: affinity(%[[AFFINITY]])
    %cmd = hal.command_buffer.create device(%device : !hal.device)
                                       mode(None)
                                 categories("Transfer|Dispatch")
                                   affinity(%affinity) : !hal.command_buffer
    // CHECK-NEXT: hal.command_buffer.execution_barrier<%[[CMD]] : !hal.command_buffer>
    hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source(CommandIssue)
                                                                     target(CommandProcess)
                                                                      flags(None)
    // CHECK-NEXT: hal.command_buffer.finalize<%[[CMD]] : !hal.command_buffer>
    hal.command_buffer.finalize<%cmd : !hal.command_buffer>
    // CHECK-NOT: hal.return
    hal.return %cmd : !hal.command_buffer
  }
  // CHECK: util.return %[[CMD]]
  util.return %result : !hal.command_buffer
}
