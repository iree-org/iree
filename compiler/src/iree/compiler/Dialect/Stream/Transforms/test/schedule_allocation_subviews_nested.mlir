// RUN: iree-opt --iree-stream-schedule-allocation %s | FileCheck %s --implicit-check-not=stream.async. --implicit-check-not=stream.resource.subview

// Nested views compose their offsets relative to the captured source buffer.

// CHECK-LABEL: util.func public @nestedSubviews(
// CHECK-SAME: %[[SOURCE:.+]]: !stream.resource<external>, %[[TARGET:.+]]: !stream.resource<external>, %[[OUTER_OFFSET:.+]]: index, %[[INNER_OFFSET:.+]]: index)
util.func public @nestedSubviews(%source: !stream.resource<external>, %target: !stream.resource<external>, %outer_offset: index, %inner_offset: index) {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
  // CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  // CHECK: %[[OFFSET:.+]] = arith.addi %[[OUTER_OFFSET]], %[[INNER_OFFSET]] : index
  // CHECK: stream.cmd.execute with(%[[SOURCE]] as %[[SOURCE_ARG:.+]]: !stream.resource<external>{%[[C128]]}, %[[TARGET]] as %[[TARGET_ARG:.+]]: !stream.resource<external>{%[[C16]]}) {
  // CHECK-NEXT: stream.cmd.copy %[[SOURCE_ARG]][%[[OFFSET]]], %[[TARGET_ARG]][%[[C0]]], %[[C16]] : !stream.resource<external>{%[[C128]]} -> !stream.resource<external>{%[[C16]]}
  %result, %timepoint = stream.async.execute with(%source as %source_arg: !stream.resource<external>{%c128}, %target as %target_arg: !stream.resource<external>{%c16}) -> %target as !stream.resource<external>{%c16} {
    %outer = stream.resource.subview %source_arg[%outer_offset] : !stream.resource<external>{%c128} -> !stream.resource<external>{%c64}
    %inner = stream.resource.subview %outer[%inner_offset] : !stream.resource<external>{%c64} -> !stream.resource<external>{%c16}
    %updated = stream.async.update %inner, %target_arg[%c0 to %c16] : !stream.resource<external>{%c16} -> %target_arg as !stream.resource<external>{%c16}
    stream.yield %updated : !stream.resource<external>{%c16}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[TARGET]],
  util.optimization_barrier %result, %timepoint : !stream.resource<external>, !stream.timepoint
  // CHECK: util.return
  util.return
}
