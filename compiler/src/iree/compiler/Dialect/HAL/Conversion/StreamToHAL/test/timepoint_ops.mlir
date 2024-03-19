// RUN: iree-opt --split-input-file --iree-hal-conversion %s | FileCheck %s

// CHECK-LABEL: util.global private mutable @rwTimepoint : !hal.fence
util.global private mutable @rwTimepoint = #stream.timepoint<immediate>
// CHECK: util.func public @globalTimepoint(%arg0: !hal.fence) -> !hal.fence
util.func public @globalTimepoint(%arg0: !stream.timepoint) -> !stream.timepoint {
  // CHECK: util.global.store %arg0, @rwTimepoint
  util.global.store %arg0, @rwTimepoint : !stream.timepoint
  // CHECK: %[[VALUE:.+]] = util.global.load @rwTimepoint
  %value = util.global.load @rwTimepoint : !stream.timepoint
  // CHECK: util.return %[[VALUE]]
  util.return %value : !stream.timepoint
}

// -----

// CHECK-LABEL: @timepointImmediate
util.func public @timepointImmediate() -> !stream.timepoint {
  // CHECK: %[[FENCE:.+]] = util.null : !hal.fence
  %0 = stream.timepoint.immediate => !stream.timepoint
  // CHECK: util.return %[[FENCE]]
  util.return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @timepointImportFence
util.func public @timepointImportFence(%arg0: !hal.fence) -> !stream.timepoint {
  %0 = stream.timepoint.import %arg0 : (!hal.fence) => !stream.timepoint
  // CHECK: util.return %arg0
  util.return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @timepointExportFence
util.func public @timepointExportFence(%arg0: !stream.timepoint) -> !hal.fence {
  %0 = stream.timepoint.export %arg0 => (!hal.fence)
  // CHECK: util.return %arg0
  util.return %0 : !hal.fence
}

// -----

// CHECK-LABEL: @timepointChainExternal
//  CHECK-SAME: (%[[TIMEPOINT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence)
util.func public @timepointChainExternal(%timepoint: !stream.timepoint, %signal: !hal.fence) {
  // CHECK: %[[DEVICE:.+]] = hal.devices.get %{{.+}}
  // CHECK: hal.device.queue.execute<%[[DEVICE]] : !hal.device> affinity(%c-1_i64) wait(%[[TIMEPOINT]]) signal(%[[SIGNAL]])
  stream.timepoint.chain_external %timepoint => (%signal : !hal.fence)
  util.return
}

// -----

// CHECK-LABEL: @timepointJoin
util.func public @timepointJoin(%arg0: !stream.timepoint, %arg1: !stream.timepoint) -> !stream.timepoint {
  // CHECK: %[[FENCE:.+]] = hal.fence.join at([%arg0, %arg1]) -> !hal.fence
  %0 = stream.timepoint.join max(%arg0, %arg1) => !stream.timepoint
  // CHECK: util.return %[[FENCE]]
  util.return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @timepointBarrier
//  CHECK-SAME: (%[[R0:.+]]: !hal.buffer) -> (!hal.buffer, !hal.fence)
util.func public @timepointBarrier(%r0: !stream.resource<external>) -> (!stream.resource<external>, !stream.timepoint) {
  %c128 = arith.constant 128 : index
  // CHECK: %[[R1T:.+]] = util.null : !hal.fence
  %r1, %r1t = stream.timepoint.barrier %r0 : !stream.resource<external>{%c128} => !stream.timepoint
  // CHECK: util.return %[[R0]], %[[R1T]]
  util.return %r1, %r1t : !stream.resource<external>, !stream.timepoint
}

// -----

// CHECK-LABEL: @timepointAwait
util.func public @timepointAwait(%arg0: !stream.timepoint, %arg1: !stream.resource<staging>, %arg2: !stream.resource<*>) -> (!stream.resource<staging>, !stream.resource<*>) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK: %[[WAIT_OK:.+]] = hal.fence.await until([%arg0]) timeout_millis(%c-1_i32) : i32
  // CHECK-NEXT: util.status.check_ok %[[WAIT_OK]]
  %0:2 = stream.timepoint.await %arg0 => %arg1, %arg2 : !stream.resource<staging>{%c100}, !stream.resource<*>{%c200}
  // CHECK: util.return %arg1, %arg2
  util.return %0#0, %0#1 : !stream.resource<staging>, !stream.resource<*>
}
