// RUN: iree-opt --split-input-file --iree-stream-materialize-builtins %s | FileCheck %s --check-prefixes=CHECK,NATIVE
// RUN: iree-opt --split-input-file --iree-stream-materialize-builtins --iree-stream-emulate-memset %s | FileCheck %s --check-prefixes=CHECK,EMULATED

// Tests that i8 splats are preserved in native mode and emulated when the flag
// is set.

// NATIVE-LABEL: @splatI32
// EMULATED-LABEL: stream.executable private @__builtin_splat_i32
// EMULATED: @splatI32
util.func public @splatI32(%arg0: index, %arg1: i32) -> !stream.resource<*> {
  // NATIVE: %[[RET:.+]] = stream.async.splat %arg1
  // EMULATED: %[[COUNT:.+]] = arith.divui %arg0, %c4
  // EMULATED: %[[RET:.+]] = stream.async.dispatch @__builtin_splat_i32::@__builtin_splat_i32[%[[COUNT]]](%arg1, %[[COUNT]]) : (i32, index) -> !stream.resource<*>{%arg0}
  %0 = stream.async.splat %arg1 : i32 -> !stream.resource<*>{%arg0}
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<*>
}

// NATIVE-NOT: stream.executable private @__builtin_splat_i32

// -----

// Tests expansion of the stream.async.splat op for i64 types.

// CHECK-LABEL: stream.executable private @__builtin_splat_i64
// CHECK: @builtinSplatI64
util.func public @builtinSplatI64(%arg0: index, %arg1: i64) -> !stream.resource<*> {
  // CHECK: %[[COUNT:.+]] = arith.divui %arg0, %c8
  // CHECK: %[[RET:.+]] = stream.async.dispatch @__builtin_splat_i64::@__builtin_splat_i64[%[[COUNT]]](%arg1, %[[COUNT]]) : (i64, index) -> !stream.resource<*>{%arg0}
  %0 = stream.async.splat %arg1 : i64 -> !stream.resource<*>{%arg0}
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<*>
}

// -----

// Tests expansion of the stream.async.fill op for i64 types.

// CHECK-LABEL: stream.executable private @__builtin_fill_i64
// CHECK: @builtinFillI64
// CHECK-SAME: (%[[RES:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[VALUE:.+]]: i64, %[[BYTE_OFFSET:.+]]: index, %[[BYTE_END:.+]]: index, %[[BYTE_LENGTH:.+]]: index)
util.func public @builtinFillI64(%res: !stream.resource<*>, %size: index, %value: i64, %byte_offset: index, %byte_end: index, %byte_length: index) -> !stream.resource<*> {
  // CHECK: %[[COUNT:.+]] = arith.divui %[[BYTE_LENGTH]], %c8
  // CHECK: %[[RET:.+]] = stream.async.dispatch @__builtin_fill_i64::@__builtin_fill_i64[%[[COUNT]]](%[[RES]][%[[BYTE_OFFSET]] to %[[BYTE_END]] for %[[BYTE_LENGTH]]], %[[VALUE]], %[[COUNT]]) : (!stream.resource<*>{%[[SIZE]]}, i64, index) -> %[[RES]]{%[[SIZE]]}
  %0 = stream.async.fill %value, %res[%byte_offset to %byte_end for %byte_length] : i64 -> %arg0 as !stream.resource<*>{%size}
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<*>
}

// -----

// Tests that builtins nested in execution regions have ops inserted in the
// correct places.

// CHECK-LABEL: stream.executable private @__builtin_splat_i64
// CHECK: @builtinSplatI64
util.func public @builtinSplatI64(%arg0: index, %arg1: i64) -> (!stream.resource<*>, !stream.timepoint) {
  // CHECK: %[[COUNT:.+]] = arith.divui %arg0, %c8
  // CHECK: = stream.async.execute
  %0:2 = stream.async.execute with() -> !stream.resource<*>{%arg0} {
    // CHECK: stream.async.concurrent
    %1 = stream.async.concurrent with() -> !stream.resource<*>{%arg0} {
      // CHECK: %[[SPLAT:.+]] = stream.async.dispatch @__builtin_splat_i64::@__builtin_splat_i64[%[[COUNT]]](%arg1, %[[COUNT]]) : (i64, index) -> !stream.resource<*>{%arg0}
      %2 = stream.async.splat %arg1 : i64 -> !stream.resource<*>{%arg0}
      // CHECK: stream.yield %[[SPLAT]]
      stream.yield %2 : !stream.resource<*>{%arg0}
    }
    stream.yield %1 : !stream.resource<*>{%arg0}
  } => !stream.timepoint
  util.return %0#0, %0#1 : !stream.resource<*>, !stream.timepoint
}

// -----

// Tests that builtins used in multiple functions share the same executable.

// CHECK-LABEL: stream.executable private @__builtin_splat_i64
// CHECK: util.initializer
util.initializer {
  %c128 = arith.constant 128 : index
  %c0_i64 = arith.constant 0 : i64
  // CHECK: = stream.async.dispatch @__builtin_splat_i64::@__builtin_splat_i64
  %0 = stream.async.splat %c0_i64 : i64 -> !stream.resource<*>{%c128}
  util.return
}

// CHECK: util.func public @otherUser
util.func public @otherUser() -> !stream.resource<*> {
  %c128 = arith.constant 128 : index
  %c1_i64 = arith.constant 1 : i64
  // CHECK: %[[RET:.+]] = stream.async.dispatch @__builtin_splat_i64::@__builtin_splat_i64
  %0 = stream.async.splat %c1_i64 : i64 -> !stream.resource<*>{%c128}
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<*>
}

// CHECK-NOT: stream.executable private @__builtin_splat_i64
