// RUN: iree-opt --iree-stream-schedule-allocation %s | FileCheck %s --implicit-check-not=stream.async. --implicit-check-not=stream.resource.subview

// Overlapping temporary buffers share an allocation with distinct packed
// offsets. Views add their local offsets to their own buffer's packed offset.

// CHECK-LABEL: util.func public @transientSubviews(
// CHECK-SAME: %[[TARGET:.+]]: !stream.resource<external>)
util.func public @transientSubviews(%target: !stream.resource<external>) {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
  // CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
  // CHECK-DAG: %[[C64:.+]] = arith.constant 64 : index
  // CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
  // CHECK-DAG: %[[ONE:.+]] = arith.constant 1 : i8
  // CHECK-DAG: %[[TWO:.+]] = arith.constant 2 : i8
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c1_i8 = arith.constant 1 : i8
  %c2_i8 = arith.constant 2 : i8
  // CHECK: %[[PACK:.+]]:3 = stream.resource.pack slices({
  // CHECK-NEXT: [{{.+}}] = %[[C128]],
  // CHECK-NEXT: [{{.+}}] = %[[C128]]
  // CHECK: %[[SLAB:.+]], %[[READY:.+]] = stream.resource.alloca {{.*}}!stream.resource<transient>{%[[PACK]]#0}
  // CHECK-DAG: %[[FIRST_OFFSET:.+]] = arith.addi %[[PACK]]#1, %[[C32]] : index
  // CHECK-DAG: %[[SECOND_OFFSET:.+]] = arith.addi %[[PACK]]#2, %[[C16]] : index
  // CHECK: stream.cmd.execute await(%[[READY]]) => with(%[[TARGET]] as %[[DEST:.+]]: !stream.resource<external>{%[[C64]]}, %[[SLAB]] as %[[SOURCE:.+]]: !stream.resource<transient>{%[[PACK]]#0}) {
  // CHECK-NEXT: stream.cmd.fill %[[ONE]], %[[SOURCE]][%[[PACK]]#1 for %[[C128]]] : i8 -> !stream.resource<transient>{%[[PACK]]#0}
  // CHECK-NEXT: stream.cmd.fill %[[TWO]], %[[SOURCE]][%[[PACK]]#2 for %[[C128]]] : i8 -> !stream.resource<transient>{%[[PACK]]#0}
  // CHECK-NEXT: stream.cmd.copy %[[SOURCE]][%[[FIRST_OFFSET]]], %[[DEST]][%[[C0]]], %[[C32]] : !stream.resource<transient>{%[[PACK]]#0} -> !stream.resource<external>{%[[C64]]}
  // CHECK-NEXT: stream.cmd.copy %[[SOURCE]][%[[SECOND_OFFSET]]], %[[DEST]][%[[C32]]], %[[C32]] : !stream.resource<transient>{%[[PACK]]#0} -> !stream.resource<external>{%[[C64]]}
  %result, %timepoint = stream.async.execute with(%target as %target_arg: !stream.resource<external>{%c64}) -> %target as !stream.resource<external>{%c64} {
    %first = stream.async.splat %c1_i8 : i8 -> !stream.resource<transient>{%c128}
    %second = stream.async.splat %c2_i8 : i8 -> !stream.resource<transient>{%c128}
    %first_view = stream.resource.subview %first[%c32] : !stream.resource<transient>{%c128} -> !stream.resource<transient>{%c32}
    %second_view = stream.resource.subview %second[%c16] : !stream.resource<transient>{%c128} -> !stream.resource<transient>{%c32}
    %updated = stream.async.update %first_view, %target_arg[%c0 to %c32] : !stream.resource<transient>{%c32} -> %target_arg as !stream.resource<external>{%c64}
    %result = stream.async.update %second_view, %updated[%c32 to %c64] : !stream.resource<transient>{%c32} -> %updated as !stream.resource<external>{%c64}
    stream.yield %result : !stream.resource<external>{%c64}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[TARGET]],
  util.optimization_barrier %result, %timepoint : !stream.resource<external>, !stream.timepoint
  // CHECK: util.return
  util.return
}
