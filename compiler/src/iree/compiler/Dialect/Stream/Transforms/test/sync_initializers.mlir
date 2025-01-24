// RUN: iree-opt --split-input-file --iree-stream-sync-initializers %s | FileCheck %s

util.global private @global : !stream.resource<transient>
util.initializer {
  %size = arith.constant 100 : index
  %pending_result = stream.async.alloca : !stream.resource<transient>{%size}
  %barrier_result, %barrier_timepoint = stream.timepoint.barrier %pending_result : !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: stream.timepoint.await sync
  %ready_result = stream.timepoint.await %barrier_timepoint => %barrier_result : !stream.resource<transient>{%size}
  util.global.store %ready_result, @global : !stream.resource<transient>
  util.return
}
