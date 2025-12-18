// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @FoldParameterLoadTargetSubview
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[OFFSET0:.+]]: index, %[[LENGTH0:.+]]: index, %[[OFFSET1:.+]]: index, %[[LENGTH1:.+]]: index)
util.func private @FoldParameterLoadTargetSubview(%wait: !stream.timepoint, %offset0: index, %length0: index, %offset1: index, %length1: index) -> (!stream.resource<constant>, !stream.resource<constant>, !stream.timepoint) {
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 50 : i64
  %c50_i64 = arith.constant 50 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 51 : i64
  %c51_i64 = arith.constant 51 : i64
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-DAG: %[[OFFSET0_I64:.+]] = arith.index_cast %[[OFFSET0]] : index to i64
  // CHECK-DAG: %[[FOLDED_PARAM_OFFSET0:.+]] = arith.addi %[[OFFSET0_I64]], %[[PARAM_OFFSET0]]
  // CHECK-DAG: %[[OFFSET1_I64:.+]] = arith.index_cast %[[OFFSET1]] : index to i64
  // CHECK-DAG: %[[FOLDED_PARAM_OFFSET1:.+]] = arith.addi %[[OFFSET1_I64]], %[[PARAM_OFFSET1]]
  // CHECK: %[[RESULTS:.+]]:2, %[[SIGNAL:.+]] = stream.cmd.parameter.load await(%[[WAIT]]) => {
  // CHECK-NEXT: "scope"::"key0"[%[[FOLDED_PARAM_OFFSET0]]] : !stream.resource<constant>{%[[LENGTH0]]},
  // CHECK-NEXT: "scope"::"key1"[%[[FOLDED_PARAM_OFFSET1]]] : !stream.resource<constant>{%[[LENGTH1]]}
  // CHECK-NEXT: } => !stream.timepoint
  %results:2, %result_timepoint = stream.cmd.parameter.load await(%wait) => {
    "scope"::"key0"[%c50_i64] : !stream.resource<constant>{%c100},
    "scope"::"key1"[%c51_i64] : !stream.resource<constant>{%c200}
  } => !stream.timepoint
  // CHECK-NOT: stream.resource.subview
  %subview0 = stream.resource.subview %results#0[%offset0] : !stream.resource<constant>{%c100} -> !stream.resource<constant>{%length0}
  // CHECK-NOT: stream.resource.subview
  %subview1 = stream.resource.subview %results#1[%offset1] : !stream.resource<constant>{%c200} -> !stream.resource<constant>{%length1}
  // CHECK: util.return %[[RESULTS]]#0, %[[RESULTS]]#1, %[[SIGNAL]]
  util.return %subview0, %subview1, %result_timepoint : !stream.resource<constant>, !stream.resource<constant>, !stream.timepoint
}

// -----

// CHECK-LABEL: @FoldParameterReadTargetSubview
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[TARGET:.+]]: !stream.resource<transient>, %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index)
util.func private @FoldParameterReadTargetSubview(%wait: !stream.timepoint, %target: !stream.resource<transient>, %offset: index, %length: index) -> !stream.timepoint {
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 50 : i64
  %c50_i64 = arith.constant 50 : i64
  // CHECK-DAG: %[[RESOURCE_OFFSET:.+]] = arith.constant 100 : index
  %c100 = arith.constant 100 : index
  // CHECK-DAG: %[[TRANSFER_LENGTH:.+]] = arith.constant 200 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[OFFSET]] : index to i64
  // CHECK-DAG: %[[FOLDED_PARAM_OFFSET:.+]] = arith.addi %[[OFFSET_I64]], %[[PARAM_OFFSET]]
  // CHECK-DAG: %[[FOLDED_RESOURCE_OFFSET:.+]] = arith.addi %[[OFFSET]], %[[RESOURCE_OFFSET]]
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %target[%offset] : !stream.resource<transient>{%length} -> !stream.resource<transient>{%c300}
  // CHECK: = stream.cmd.parameter.read await(%[[WAIT]]) => "scope"::"key"[%[[FOLDED_PARAM_OFFSET]]] -> %[[TARGET]][%[[FOLDED_RESOURCE_OFFSET]] for %[[TRANSFER_LENGTH]]] : !stream.resource<transient>{%[[LENGTH]]} => !stream.timepoint
  %timepoint = stream.cmd.parameter.read await(%wait) => "scope"::"key"[%c50_i64] -> %subview[%c100 for %c200] : !stream.resource<transient>{%c300} => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}

// -----

// CHECK-LABEL: @FoldParameterWriteSourceSubview
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[SOURCE:.+]]: !stream.resource<transient>, %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index)
util.func private @FoldParameterWriteSourceSubview(%wait: !stream.timepoint, %source: !stream.resource<transient>, %offset: index, %length: index) -> !stream.timepoint {
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 50 : i64
  %c50_i64 = arith.constant 50 : i64
  // CHECK-DAG: %[[RESOURCE_OFFSET:.+]] = arith.constant 100 : index
  %c100 = arith.constant 100 : index
  // CHECK-DAG: %[[TRANSFER_LENGTH:.+]] = arith.constant 200 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[OFFSET]] : index to i64
  // CHECK-DAG: %[[FOLDED_PARAM_OFFSET:.+]] = arith.addi %[[OFFSET_I64]], %[[PARAM_OFFSET]]
  // CHECK-DAG: %[[FOLDED_RESOURCE_OFFSET:.+]] = arith.addi %[[OFFSET]], %[[RESOURCE_OFFSET]]
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %source[%offset] : !stream.resource<transient>{%length} -> !stream.resource<transient>{%c300}
  // CHECK: = stream.cmd.parameter.write await(%[[WAIT]]) => %[[SOURCE]][%[[FOLDED_RESOURCE_OFFSET]] for %[[TRANSFER_LENGTH]]] : !stream.resource<transient>{%[[LENGTH]]} -> "scope"::"key"[%[[FOLDED_PARAM_OFFSET]]] => !stream.timepoint
  %timepoint = stream.cmd.parameter.write await(%wait) => %subview[%c100 for %c200] : !stream.resource<transient>{%c300} -> "scope"::"key"[%c50_i64] => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}

// -----

// CHECK-LABEL: @FoldParameterGatherTargetSubview
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[TARGET:.+]]: !stream.resource<transient>, %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index)
util.func private @FoldParameterGatherTargetSubview(%wait: !stream.timepoint, %target: !stream.resource<transient>, %offset: index, %length: index) -> !stream.timepoint {
  %c50_i64 = arith.constant 50 : i64
  %c51_i64 = arith.constant 51 : i64
  %c100 = arith.constant 100 : index
  %c101 = arith.constant 101 : index
  %c200 = arith.constant 200 : index
  %c201 = arith.constant 201 : index
  %c300 = arith.constant 300 : index
  // CHECK-DAG: %[[RESOURCE_OFFSET0:.+]] = arith.addi %[[OFFSET]], %c100
  // CHECK-DAG: %[[RESOURCE_OFFSET1:.+]] = arith.addi %[[OFFSET]], %c101
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %target[%offset] : !stream.resource<transient>{%length} -> !stream.resource<transient>{%c300}
  // CHECK: %{{.+}} = stream.cmd.parameter.gather await(%[[WAIT]]) => {
  // CHECK-NEXT: "scope"::"key0"[%c50_i64] -> %[[TARGET]][%[[RESOURCE_OFFSET0]] for %c200] : !stream.resource<transient>{%[[LENGTH]]},
  // CHECK-NEXT: "scope"::"key1"[%c51_i64] -> %[[TARGET]][%[[RESOURCE_OFFSET1]] for %c201] : !stream.resource<transient>{%[[LENGTH]]}
  // CHECK-NEXT: } => !stream.timepoint
  %timepoint = stream.cmd.parameter.gather await(%wait) => {
    "scope"::"key0"[%c50_i64] -> %subview[%c100 for %c200] : !stream.resource<transient>{%c300},
    "scope"::"key1"[%c51_i64] -> %subview[%c101 for %c201] : !stream.resource<transient>{%c300}
  } => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}

// -----

// CHECK-LABEL: @FoldParameterScatterSourceSubview
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[SOURCE:.+]]: !stream.resource<transient>, %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index)
util.func private @FoldParameterScatterSourceSubview(%wait: !stream.timepoint, %source: !stream.resource<transient>, %offset: index, %length: index) -> !stream.timepoint {
  %c50_i64 = arith.constant 50 : i64
  %c51_i64 = arith.constant 51 : i64
  %c100 = arith.constant 100 : index
  %c101 = arith.constant 101 : index
  %c200 = arith.constant 200 : index
  %c201 = arith.constant 201 : index
  %c300 = arith.constant 300 : index
  // CHECK-DAG: %[[RESOURCE_OFFSET0:.+]] = arith.addi %[[OFFSET]], %c100
  // CHECK-DAG: %[[RESOURCE_OFFSET1:.+]] = arith.addi %[[OFFSET]], %c101
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %source[%offset] : !stream.resource<transient>{%length} -> !stream.resource<transient>{%c300}
  // CHECK: %{{.+}} = stream.cmd.parameter.scatter await(%[[WAIT]]) => {
  // CHECK-NEXT: %[[SOURCE]][%[[RESOURCE_OFFSET0]] for %c200] : !stream.resource<transient>{%[[LENGTH]]} -> "scope"::"key0"[%c50_i64],
  // CHECK-NEXT: %[[SOURCE]][%[[RESOURCE_OFFSET1]] for %c201] : !stream.resource<transient>{%[[LENGTH]]} -> "scope"::"key1"[%c51_i64]
  // CHECK-NEXT: } => !stream.timepoint
  %timepoint = stream.cmd.parameter.scatter await(%wait) => {
    %subview[%c100 for %c200] : !stream.resource<transient>{%c300} -> "scope"::"key0"[%c50_i64],
    %subview[%c101 for %c201] : !stream.resource<transient>{%c300} -> "scope"::"key1"[%c51_i64]
  } => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}
