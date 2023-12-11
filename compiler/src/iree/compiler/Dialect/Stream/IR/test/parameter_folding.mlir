// RUN: iree-opt --split-input-file --canonicalize %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @FoldParameterLoadTargetSubview
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[OFFSET0:.+]]: index, %[[LENGTH0:.+]]: index, %[[OFFSET1:.+]]: index, %[[LENGTH1:.+]]: index)
func.func @FoldParameterLoadTargetSubview(%wait: !stream.timepoint, %offset0: index, %length0: index, %offset1: index, %length1: index) -> (!stream.resource<constant>, !stream.resource<constant>, !stream.timepoint) {
  %c50_i64 = arith.constant 50 : i64
  %c51_i64 = arith.constant 51 : i64
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-DAG: %[[OFFSET0_I64:.+]] = arith.index_cast %[[OFFSET0]] : index to i64
  // CHECK-DAG: %[[PARAMETER_OFFSET0:.+]] = arith.addi %[[OFFSET0_I64]], %c50_i64
  // CHECK-DAG: %[[OFFSET1_I64:.+]] = arith.index_cast %[[OFFSET1]] : index to i64
  // CHECK-DAG: %[[PARAMETER_OFFSET1:.+]] = arith.addi %[[OFFSET1_I64]], %c51_i64
  // CHECK: %[[RESULTS:.+]]:2, %[[SIGNAL:.+]] = stream.parameter.load await(%[[WAIT]]) => {
  // CHECK-NEXT: "scope"::"key0"[%[[PARAMETER_OFFSET0]]] : !stream.resource<constant>{%[[LENGTH0]]},
  // CHECK-NEXT: "scope"::"key1"[%[[PARAMETER_OFFSET1]]] : !stream.resource<constant>{%[[LENGTH1]]}
  // CHECK-NEXT: } => !stream.timepoint
  %results:2, %result_timepoint = stream.parameter.load await(%wait) => {
    "scope"::"key0"[%c50_i64] : !stream.resource<constant>{%c100},
    "scope"::"key1"[%c51_i64] : !stream.resource<constant>{%c200}
  } => !stream.timepoint
  // CHECK-NOT: stream.resource.subview
  %subview0 = stream.resource.subview %results#0[%offset0] : !stream.resource<constant>{%c100} -> !stream.resource<constant>{%length0}
  // CHECK-NOT: stream.resource.subview
  %subview1 = stream.resource.subview %results#1[%offset1] : !stream.resource<constant>{%c200} -> !stream.resource<constant>{%length1}
  // CHECK: return %[[RESULTS]]#0, %[[RESULTS]]#1, %[[SIGNAL]]
  return %subview0, %subview1, %result_timepoint : !stream.resource<constant>, !stream.resource<constant>, !stream.timepoint
}

// -----

// CHECK-LABEL: @FoldParameterReadTargetSubview
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[TARGET:.+]]: !stream.resource<transient>, %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index)
func.func @FoldParameterReadTargetSubview(%wait: !stream.timepoint, %target: !stream.resource<transient>, %offset: index, %length: index) -> !stream.timepoint {
  %c50_i64 = arith.constant 50 : i64
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[OFFSET]] : index to i64
  // CHECK-DAG: %[[PARAMETER_OFFSET:.+]] = arith.addi %[[OFFSET_I64]], %c50_i64
  // CHECK-DAG: %[[RESOURCE_OFFSET:.+]] = arith.addi %[[OFFSET]], %c100
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %target[%offset] : !stream.resource<transient>{%length} -> !stream.resource<transient>{%c300}
  // CHECK: = stream.parameter.read await(%[[WAIT]]) => "scope"::"key"[%[[PARAMETER_OFFSET]]] -> %[[TARGET]][%[[RESOURCE_OFFSET]] for %c200] : !stream.resource<transient>{%[[LENGTH]]} => !stream.timepoint
  %timepoint = stream.parameter.read await(%wait) => "scope"::"key"[%c50_i64] -> %subview[%c100 for %c200] : !stream.resource<transient>{%c300} => !stream.timepoint
  return %timepoint : !stream.timepoint
}

// -----

// CHECK-LABEL: @FoldParameterWriteSourceSubview
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[SOURCE:.+]]: !stream.resource<transient>, %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index)
func.func @FoldParameterWriteSourceSubview(%wait: !stream.timepoint, %source: !stream.resource<transient>, %offset: index, %length: index) -> !stream.timepoint {
  %c50_i64 = arith.constant 50 : i64
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[OFFSET]] : index to i64
  // CHECK-DAG: %[[PARAMETER_OFFSET:.+]] = arith.addi %[[OFFSET_I64]], %c50_i64
  // CHECK-DAG: %[[RESOURCE_OFFSET:.+]] = arith.addi %[[OFFSET]], %c100
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %source[%offset] : !stream.resource<transient>{%length} -> !stream.resource<transient>{%c300}
  // CHECK: = stream.parameter.write await(%[[WAIT]]) => %[[SOURCE]][%[[RESOURCE_OFFSET]] for %c200] : !stream.resource<transient>{%[[LENGTH]]} -> "scope"::"key"[%[[PARAMETER_OFFSET]]] => !stream.timepoint
  %timepoint = stream.parameter.write await(%wait) => %subview[%c100 for %c200] : !stream.resource<transient>{%c300} -> "scope"::"key"[%c50_i64] => !stream.timepoint
  return %timepoint : !stream.timepoint
}
