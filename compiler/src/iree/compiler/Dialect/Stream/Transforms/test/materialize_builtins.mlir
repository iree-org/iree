// RUN: iree-opt --split-input-file --iree-stream-materialize-builtins %s | FileCheck %s

// Tests expansion of the stream.builtin.splat.i64 op.

// CHECK-LABEL: @builtinSplatI64
func.func @builtinSplatI64(%arg0: index, %arg1: i64) -> !stream.resource<*> {
  // CHECK: %[[COUNT:.+]] = arith.divui %arg0, %c8
  // CHECK: %[[RET:.+]] = stream.async.dispatch @__builtin_splat_i64::@__builtin_splat_i64[%[[COUNT]]](%arg1, %[[COUNT]]) : (i64, index) -> !stream.resource<*>{%arg0}
  %0 = stream.builtin.splat.i64 %arg1 : i64 -> !stream.resource<*>{%arg0}
  // CHECK: return %[[RET]]
  return %0 : !stream.resource<*>
}

// CHECK: stream.executable private @__builtin_splat_i64

// -----

// Tests expansion of the stream.builtin.fill.i64 op.

// CHECK-LABEL: @builtinFillI64
// CHECK-SAME: (%[[RES:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[VALUE:.+]]: i64, %[[BYTE_OFFSET:.+]]: index, %[[BYTE_END:.+]]: index, %[[BYTE_LENGTH:.+]]: index)
func.func @builtinFillI64(%res: !stream.resource<*>, %size: index, %value: i64, %byte_offset: index, %byte_end: index, %byte_length: index) -> !stream.resource<*> {
  // CHECK: %[[COUNT:.+]] = arith.divui %[[BYTE_LENGTH]], %c8
  // CHECK: %[[RET:.+]] = stream.async.dispatch @__builtin_fill_i64::@__builtin_fill_i64[%[[COUNT]]](%[[RES]][%[[BYTE_OFFSET]] to %[[BYTE_END]] for %[[BYTE_LENGTH]]], %[[VALUE]], %[[BYTE_OFFSET]], %[[COUNT]]) : (!stream.resource<*>{%[[SIZE]]}, i64, index, index) -> %[[RES]]{%[[SIZE]]}
  %0 = stream.builtin.fill.i64 %value, %res[%byte_offset to %byte_end for %byte_length] : i64 -> %arg0 as !stream.resource<*>{%size}
  // CHECK: return %[[RET]]
  return %0 : !stream.resource<*>
}

// CHECK: stream.executable private @__builtin_fill_i64

// -----

// Tests that builtins used in multiple functions share the same executable.

// CHECK: util.initializer
util.initializer {
  %c128 = arith.constant 128 : index
  %c0_i64 = arith.constant 0 : i64
  // CHECK: = stream.async.dispatch @__builtin_splat_i64::@__builtin_splat_i64
  %0 = stream.builtin.splat.i64 %c0_i64 : i64 -> !stream.resource<*>{%c128}
  util.initializer.return
}

// CHECK: stream.executable private @__builtin_splat_i64

// CHECK: func.func @otherUser
func.func @otherUser() -> !stream.resource<*> {
  %c128 = arith.constant 128 : index
  %c1_i64 = arith.constant 1 : i64
  // CHECK: %[[RET:.+]] = stream.async.dispatch @__builtin_splat_i64::@__builtin_splat_i64
  %0 = stream.builtin.splat.i64 %c1_i64 : i64 -> !stream.resource<*>{%c128}
  // CHECK: return %[[RET]]
  return %0 : !stream.resource<*>
}

// CHECK-NOT: stream.executable private @__builtin_splat_i64
