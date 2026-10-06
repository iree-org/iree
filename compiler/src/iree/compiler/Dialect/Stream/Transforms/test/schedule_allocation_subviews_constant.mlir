// RUN: iree-opt --iree-stream-schedule-allocation %s | FileCheck %s --implicit-check-not=stream.async. --implicit-check-not=stream.resource.subview

// An extracted constant can both escape the execution region and supply a
// subview to a copy within it. The copy uses the captured constant buffer.

// CHECK-LABEL: util.func public @constantSubview(
// CHECK-SAME: %[[TARGET:.+]]: !stream.resource<external>)
util.func public @constantSubview(%target: !stream.resource<external>) {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
  // CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
  // CHECK-DAG: %[[C64:.+]] = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  // CHECK: %[[CONSTANT:.+]], %[[READY:.+]] = stream.resource.constants :
  // CHECK-NEXT: !stream.resource<constant>{%[[C64]]} = #stream.parameter.named<"weights"::"table"> : tensor<64xi8>
  // CHECK: stream.cmd.execute await(%[[READY]]) => with(%[[TARGET]] as %[[DEST:.+]]: !stream.resource<external>{%[[C32]]}, %[[CONSTANT]] as %[[SOURCE:.+]]: !stream.resource<constant>{%[[C64]]}) {
  // CHECK-NEXT: stream.cmd.copy %[[SOURCE]][%[[C16]]], %[[DEST]][%[[C0]]], %[[C32]] : !stream.resource<constant>{%[[C64]]} -> !stream.resource<external>{%[[C32]]}
  %results:2, %timepoint = stream.async.execute with(%target as %target_arg: !stream.resource<external>{%c32}) -> (!stream.resource<constant>{%c64}, %target as !stream.resource<external>{%c32}) {
    %constant = stream.async.constant : !stream.resource<constant>{%c64} =
      #stream.parameter.named<"weights"::"table"> : tensor<64xi8>
    %view = stream.resource.subview %constant[%c16] : !stream.resource<constant>{%c64} -> !stream.resource<constant>{%c32}
    %updated = stream.async.update %view, %target_arg[%c0 to %c32] : !stream.resource<constant>{%c32} -> %target_arg as !stream.resource<external>{%c32}
    stream.yield %constant, %updated : !stream.resource<constant>{%c64}, !stream.resource<external>{%c32}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[CONSTANT]], %[[TARGET]],
  util.optimization_barrier %results#0, %results#1, %timepoint : !stream.resource<constant>, !stream.resource<external>, !stream.timepoint
  // CHECK: util.return
  util.return
}
