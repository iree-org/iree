// RUN: iree-opt --split-input-file --iree-hal-conversion %s | FileCheck %s

// NOTE: timepoints are tested in StreamToHAL/test/timepoint_ops.mlir because
// their usage is more tied to the ops loading/storing the globals. This file
// just tests the generic !stream.* type globals.

// CHECK: util.global private mutable @resource : !hal.buffer
util.global private mutable @resource : !stream.resource<variable>

// CHECK-LABEL: @resourceGlobals
// CHECK-SAME: (%[[ARG0:.+]]: !hal.buffer) -> !hal.buffer
util.func private @resourceGlobals(%arg0: !stream.resource<variable>) -> !stream.resource<variable> {
  // CHECK: util.global.store %[[ARG0]], @resource : !hal.buffer
  util.global.store %arg0, @resource : !stream.resource<variable>
  // CHECK: %[[VALUE:.+]] = util.global.load @resource : !hal.buffer
  %value = util.global.load @resource : !stream.resource<variable>
  // CHECK: util.return %[[VALUE]]
  util.return %value : !stream.resource<variable>
}
