// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @FoldParameterLoadTargetSubview
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[OFFSET0:.+]]: index, %[[LENGTH0:.+]]: index, %[[OFFSET1:.+]]: index, %[[LENGTH1:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer)
util.func private @FoldParameterLoadTargetSubview(%wait: !stream.timepoint, %offset0: index, %length0: index, %offset1: index, %length1: index, %scope: !util.buffer, %key0: !util.buffer, %key1: !util.buffer) -> (!stream.resource<constant>, !stream.resource<constant>, !stream.timepoint) {
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 50 : i64
  %param_offset0 = arith.constant 50 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 51 : i64
  %param_offset1 = arith.constant 51 : i64
  %size0 = arith.constant 100 : index
  %size1 = arith.constant 200 : index
  // CHECK-DAG: %[[OFFSET0_I64:.+]] = arith.index_cast %[[OFFSET0]] : index to i64
  // CHECK-DAG: %[[FOLDED_PARAM_OFFSET0:.+]] = arith.addi %[[OFFSET0_I64]], %[[PARAM_OFFSET0]]
  // CHECK-DAG: %[[OFFSET1_I64:.+]] = arith.index_cast %[[OFFSET1]] : index to i64
  // CHECK-DAG: %[[FOLDED_PARAM_OFFSET1:.+]] = arith.addi %[[OFFSET1_I64]], %[[PARAM_OFFSET1]]
  // CHECK: %[[RESULTS:.+]]:2, %[[SIGNAL:.+]] = stream.cmd.parameter.load await(%[[WAIT]]) => {
  // CHECK-NEXT: %[[SCOPE]]::%[[KEY0]][%[[FOLDED_PARAM_OFFSET0]]] : !stream.resource<constant>{%[[LENGTH0]]},
  // CHECK-NEXT: %[[SCOPE]]::%[[KEY1]][%[[FOLDED_PARAM_OFFSET1]]] : !stream.resource<constant>{%[[LENGTH1]]}
  // CHECK-NEXT: } => !stream.timepoint
  %results:2, %result_timepoint = stream.cmd.parameter.load await(%wait) => {
    %scope::%key0[%param_offset0] : !stream.resource<constant>{%size0},
    %scope::%key1[%param_offset1] : !stream.resource<constant>{%size1}
  } => !stream.timepoint
  // CHECK-NOT: stream.resource.subview
  %subview0 = stream.resource.subview %results#0[%offset0] : !stream.resource<constant>{%size0} -> !stream.resource<constant>{%length0}
  // CHECK-NOT: stream.resource.subview
  %subview1 = stream.resource.subview %results#1[%offset1] : !stream.resource<constant>{%size1} -> !stream.resource<constant>{%length1}
  // CHECK: util.return %[[RESULTS]]#0, %[[RESULTS]]#1, %[[SIGNAL]]
  util.return %subview0, %subview1, %result_timepoint : !stream.resource<constant>, !stream.resource<constant>, !stream.timepoint
}

// -----

// CHECK-LABEL: @FoldParameterReadTargetSubview
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[TARGET:.+]]: !stream.resource<transient>, %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func private @FoldParameterReadTargetSubview(%wait: !stream.timepoint, %target: !stream.resource<transient>, %offset: index, %length: index, %scope: !util.buffer, %key: !util.buffer) -> !stream.timepoint {
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 50 : i64
  %param_offset = arith.constant 50 : i64
  // CHECK-DAG: %[[RESOURCE_OFFSET:.+]] = arith.constant 100 : index
  %resource_offset = arith.constant 100 : index
  // CHECK-DAG: %[[TRANSFER_LENGTH:.+]] = arith.constant 200 : index
  %transfer_length = arith.constant 200 : index
  %resource_size = arith.constant 300 : index
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[OFFSET]] : index to i64
  // CHECK-DAG: %[[FOLDED_PARAM_OFFSET:.+]] = arith.addi %[[OFFSET_I64]], %[[PARAM_OFFSET]]
  // CHECK-DAG: %[[FOLDED_RESOURCE_OFFSET:.+]] = arith.addi %[[OFFSET]], %[[RESOURCE_OFFSET]]
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %target[%offset] : !stream.resource<transient>{%length} -> !stream.resource<transient>{%resource_size}
  // CHECK: = stream.cmd.parameter.read await(%[[WAIT]]) => %[[SCOPE]]::%[[KEY]][%[[FOLDED_PARAM_OFFSET]]] -> %[[TARGET]][%[[FOLDED_RESOURCE_OFFSET]] for %[[TRANSFER_LENGTH]]] : !stream.resource<transient>{%[[LENGTH]]} => !stream.timepoint
  %timepoint = stream.cmd.parameter.read await(%wait) => %scope::%key[%param_offset] -> %subview[%resource_offset for %transfer_length] : !stream.resource<transient>{%resource_size} => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}

// -----

// CHECK-LABEL: @FoldParameterWriteSourceSubview
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[SOURCE:.+]]: !stream.resource<transient>, %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func private @FoldParameterWriteSourceSubview(%wait: !stream.timepoint, %source: !stream.resource<transient>, %offset: index, %length: index, %scope: !util.buffer, %key: !util.buffer) -> !stream.timepoint {
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 50 : i64
  %param_offset = arith.constant 50 : i64
  // CHECK-DAG: %[[RESOURCE_OFFSET:.+]] = arith.constant 100 : index
  %resource_offset = arith.constant 100 : index
  // CHECK-DAG: %[[TRANSFER_LENGTH:.+]] = arith.constant 200 : index
  %transfer_length = arith.constant 200 : index
  %resource_size = arith.constant 300 : index
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[OFFSET]] : index to i64
  // CHECK-DAG: %[[FOLDED_PARAM_OFFSET:.+]] = arith.addi %[[OFFSET_I64]], %[[PARAM_OFFSET]]
  // CHECK-DAG: %[[FOLDED_RESOURCE_OFFSET:.+]] = arith.addi %[[OFFSET]], %[[RESOURCE_OFFSET]]
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %source[%offset] : !stream.resource<transient>{%length} -> !stream.resource<transient>{%resource_size}
  // CHECK: = stream.cmd.parameter.write await(%[[WAIT]]) => %[[SOURCE]][%[[FOLDED_RESOURCE_OFFSET]] for %[[TRANSFER_LENGTH]]] : !stream.resource<transient>{%[[LENGTH]]} -> %[[SCOPE]]::%[[KEY]][%[[FOLDED_PARAM_OFFSET]]] => !stream.timepoint
  %timepoint = stream.cmd.parameter.write await(%wait) => %subview[%resource_offset for %transfer_length] : !stream.resource<transient>{%resource_size} -> %scope::%key[%param_offset] => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}

// -----

// CHECK-LABEL: @FoldParameterGatherTargetSubview
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[TARGET:.+]]: !stream.resource<transient>, %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer)
util.func private @FoldParameterGatherTargetSubview(%wait: !stream.timepoint, %target: !stream.resource<transient>, %offset: index, %length: index, %scope: !util.buffer, %key0: !util.buffer, %key1: !util.buffer) -> !stream.timepoint {
  %param_offset0 = arith.constant 50 : i64
  %param_offset1 = arith.constant 51 : i64
  %target_offset0 = arith.constant 100 : index
  %target_offset1 = arith.constant 101 : index
  %transfer_length0 = arith.constant 200 : index
  %transfer_length1 = arith.constant 201 : index
  %resource_size = arith.constant 300 : index
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 50 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 51 : i64
  // CHECK-DAG: %[[TARGET_OFFSET0:.+]] = arith.constant 100 : index
  // CHECK-DAG: %[[TARGET_OFFSET1:.+]] = arith.constant 101 : index
  // CHECK-DAG: %[[TRANSFER_LENGTH0:.+]] = arith.constant 200 : index
  // CHECK-DAG: %[[TRANSFER_LENGTH1:.+]] = arith.constant 201 : index
  // CHECK-DAG: %[[FOLDED_OFFSET0:.+]] = arith.addi %[[OFFSET]], %[[TARGET_OFFSET0]]
  // CHECK-DAG: %[[FOLDED_OFFSET1:.+]] = arith.addi %[[OFFSET]], %[[TARGET_OFFSET1]]
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %target[%offset] : !stream.resource<transient>{%length} -> !stream.resource<transient>{%resource_size}
  // CHECK: %{{.+}} = stream.cmd.parameter.gather await(%[[WAIT]]) => {
  // CHECK-NEXT: %[[SCOPE]]::%[[KEY0]][%[[PARAM_OFFSET0]]] -> %[[TARGET]][%[[FOLDED_OFFSET0]] for %[[TRANSFER_LENGTH0]]] : !stream.resource<transient>{%[[LENGTH]]},
  // CHECK-NEXT: %[[SCOPE]]::%[[KEY1]][%[[PARAM_OFFSET1]]] -> %[[TARGET]][%[[FOLDED_OFFSET1]] for %[[TRANSFER_LENGTH1]]] : !stream.resource<transient>{%[[LENGTH]]}
  // CHECK-NEXT: } => !stream.timepoint
  %timepoint = stream.cmd.parameter.gather await(%wait) => {
    %scope::%key0[%param_offset0] -> %subview[%target_offset0 for %transfer_length0] : !stream.resource<transient>{%resource_size},
    %scope::%key1[%param_offset1] -> %subview[%target_offset1 for %transfer_length1] : !stream.resource<transient>{%resource_size}
  } => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}

// -----

// CHECK-LABEL: @FoldParameterScatterSourceSubview
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[SOURCE:.+]]: !stream.resource<transient>, %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer)
util.func private @FoldParameterScatterSourceSubview(%wait: !stream.timepoint, %source: !stream.resource<transient>, %offset: index, %length: index, %scope: !util.buffer, %key0: !util.buffer, %key1: !util.buffer) -> !stream.timepoint {
  %param_offset0 = arith.constant 50 : i64
  %param_offset1 = arith.constant 51 : i64
  %source_offset0 = arith.constant 100 : index
  %source_offset1 = arith.constant 101 : index
  %transfer_length0 = arith.constant 200 : index
  %transfer_length1 = arith.constant 201 : index
  %resource_size = arith.constant 300 : index
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 50 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 51 : i64
  // CHECK-DAG: %[[SOURCE_OFFSET0:.+]] = arith.constant 100 : index
  // CHECK-DAG: %[[SOURCE_OFFSET1:.+]] = arith.constant 101 : index
  // CHECK-DAG: %[[TRANSFER_LENGTH0:.+]] = arith.constant 200 : index
  // CHECK-DAG: %[[TRANSFER_LENGTH1:.+]] = arith.constant 201 : index
  // CHECK-DAG: %[[FOLDED_OFFSET0:.+]] = arith.addi %[[OFFSET]], %[[SOURCE_OFFSET0]]
  // CHECK-DAG: %[[FOLDED_OFFSET1:.+]] = arith.addi %[[OFFSET]], %[[SOURCE_OFFSET1]]
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %source[%offset] : !stream.resource<transient>{%length} -> !stream.resource<transient>{%resource_size}
  // CHECK: %{{.+}} = stream.cmd.parameter.scatter await(%[[WAIT]]) => {
  // CHECK-NEXT: %[[SOURCE]][%[[FOLDED_OFFSET0]] for %[[TRANSFER_LENGTH0]]] : !stream.resource<transient>{%[[LENGTH]]} -> %[[SCOPE]]::%[[KEY0]][%[[PARAM_OFFSET0]]],
  // CHECK-NEXT: %[[SOURCE]][%[[FOLDED_OFFSET1]] for %[[TRANSFER_LENGTH1]]] : !stream.resource<transient>{%[[LENGTH]]} -> %[[SCOPE]]::%[[KEY1]][%[[PARAM_OFFSET1]]]
  // CHECK-NEXT: } => !stream.timepoint
  %timepoint = stream.cmd.parameter.scatter await(%wait) => {
    %subview[%source_offset0 for %transfer_length0] : !stream.resource<transient>{%resource_size} -> %scope::%key0[%param_offset0],
    %subview[%source_offset1 for %transfer_length1] : !stream.resource<transient>{%resource_size} -> %scope::%key1[%param_offset1]
  } => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}
