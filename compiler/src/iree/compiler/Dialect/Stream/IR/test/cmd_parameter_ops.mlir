// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK: util.global private @parameter_unscoped = #stream.parameter.named<"key"> : tensor<10xf32>
util.global private @parameter_unscoped = #stream.parameter.named<"key"> : tensor<10xf32>
// CHECK: util.global private @parameter_scoped = #stream.parameter.named<"scope"::"key"> : tensor<10xf32>
util.global private @parameter_scoped = #stream.parameter.named<"scope"::"key"> : tensor<10xf32>
// CHECK: util.global private @parameter_config = #stream.parameter.named<"scope"::"key", {some.config = "hello"}> : tensor<10xf32>
util.global private @parameter_config = #stream.parameter.named<"scope"::"key", {some.config = "hello"}> : tensor<10xf32>

// -----

// CHECK-LABEL: @parameterLoad
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[SCOPE:.+]]: !util.buffer, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer)
util.func private @parameterLoad(%wait: !stream.timepoint, %scope: !util.buffer, %key0: !util.buffer, %key1: !util.buffer) -> (!stream.resource<constant>, !stream.resource<constant>, !stream.timepoint) {
  %param_offset0 = arith.constant 50 : i64
  %param_offset1 = arith.constant 51 : i64
  %size0 = arith.constant 100 : index
  %size1 = arith.constant 200 : index
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 50 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 51 : i64
  // CHECK-DAG: %[[SIZE0:.+]] = arith.constant 100 : index
  // CHECK-DAG: %[[SIZE1:.+]] = arith.constant 200 : index
  // CHECK: = stream.cmd.parameter.load await(%[[WAIT]]) => {
  // CHECK-NEXT: %[[SCOPE]]::%[[KEY0]][%[[PARAM_OFFSET0]]] : !stream.resource<constant>{%[[SIZE0]]},
  // CHECK-NEXT: %[[SCOPE]]::%[[KEY1]][%[[PARAM_OFFSET1]]] : !stream.resource<constant>{%[[SIZE1]]}
  // CHECK-NEXT: } => !stream.timepoint
  %results:2, %result_timepoint = stream.cmd.parameter.load await(%wait) => {
    %scope::%key0[%param_offset0] : !stream.resource<constant>{%size0},
    %scope::%key1[%param_offset1] : !stream.resource<constant>{%size1}
  } => !stream.timepoint
  util.return %results#0, %results#1, %result_timepoint : !stream.resource<constant>, !stream.resource<constant>, !stream.timepoint
}

// -----

// CHECK-LABEL: @parameterLoadNoScope
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[KEY:.+]]: !util.buffer)
util.func private @parameterLoadNoScope(%wait: !stream.timepoint, %key: !util.buffer) -> (!stream.resource<constant>, !stream.timepoint) {
  %param_offset = arith.constant 50 : i64
  %size = arith.constant 100 : index
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 50 : i64
  // CHECK-DAG: %[[SIZE:.+]] = arith.constant 100 : index
  // CHECK: = stream.cmd.parameter.load await(%[[WAIT]]) => {
  // CHECK-NEXT: %[[KEY]][%[[PARAM_OFFSET]]] : !stream.resource<constant>{%[[SIZE]]}
  // CHECK-NEXT: } => !stream.timepoint
  %result, %result_timepoint = stream.cmd.parameter.load await(%wait) => {
    %key[%param_offset] : !stream.resource<constant>{%size}
  } => !stream.timepoint
  util.return %result, %result_timepoint : !stream.resource<constant>, !stream.timepoint
}

// -----

// CHECK-LABEL: @parameterRead
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[TARGET:.+]]: !stream.resource<transient>, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func private @parameterRead(%wait: !stream.timepoint, %target: !stream.resource<transient>, %scope: !util.buffer, %key: !util.buffer) -> !stream.timepoint {
  %param_offset = arith.constant 50 : i64
  %target_offset = arith.constant 100 : index
  %length = arith.constant 200 : index
  %target_size = arith.constant 300 : index
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 50 : i64
  // CHECK-DAG: %[[TARGET_OFFSET:.+]] = arith.constant 100 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 200 : index
  // CHECK-DAG: %[[TARGET_SIZE:.+]] = arith.constant 300 : index
  // CHECK: = stream.cmd.parameter.read await(%[[WAIT]]) => %[[SCOPE]]::%[[KEY]][%[[PARAM_OFFSET]]] -> %[[TARGET]][%[[TARGET_OFFSET]] for %[[LENGTH]]] : !stream.resource<transient>{%[[TARGET_SIZE]]} => !stream.timepoint
  %timepoint = stream.cmd.parameter.read await(%wait) => %scope::%key[%param_offset] -> %target[%target_offset for %length] : !stream.resource<transient>{%target_size} => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}

// -----

// CHECK-LABEL: @parameterWrite
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[SOURCE:.+]]: !stream.resource<transient>, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func private @parameterWrite(%wait: !stream.timepoint, %source: !stream.resource<transient>, %scope: !util.buffer, %key: !util.buffer) -> !stream.timepoint {
  %param_offset = arith.constant 50 : i64
  %source_offset = arith.constant 100 : index
  %length = arith.constant 200 : index
  %source_size = arith.constant 300 : index
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 50 : i64
  // CHECK-DAG: %[[SOURCE_OFFSET:.+]] = arith.constant 100 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 200 : index
  // CHECK-DAG: %[[SOURCE_SIZE:.+]] = arith.constant 300 : index
  // CHECK: = stream.cmd.parameter.write await(%[[WAIT]]) => %[[SOURCE]][%[[SOURCE_OFFSET]] for %[[LENGTH]]] : !stream.resource<transient>{%[[SOURCE_SIZE]]} -> %[[SCOPE]]::%[[KEY]][%[[PARAM_OFFSET]]] => !stream.timepoint
  %timepoint = stream.cmd.parameter.write await(%wait) => %source[%source_offset for %length] : !stream.resource<transient>{%source_size} -> %scope::%key[%param_offset] => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}

// -----

// CHECK-LABEL: @parameterGather
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[TARGET:.+]]: !stream.resource<transient>, %[[SCOPE:.+]]: !util.buffer, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer, %[[KEY2:.+]]: !util.buffer)
util.func private @parameterGather(%wait: !stream.timepoint, %target: !stream.resource<transient>, %scope: !util.buffer, %key0: !util.buffer, %key1: !util.buffer, %key2: !util.buffer) -> !stream.timepoint {
  %param_offset0 = arith.constant 50 : i64
  %param_offset1 = arith.constant 51 : i64
  %param_offset2 = arith.constant 52 : i64
  %target_offset0 = arith.constant 100 : index
  %target_offset1 = arith.constant 101 : index
  %target_offset2 = arith.constant 102 : index
  %length0 = arith.constant 200 : index
  %length1 = arith.constant 201 : index
  %length2 = arith.constant 202 : index
  %target_size = arith.constant 300 : index
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 50 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 51 : i64
  // CHECK-DAG: %[[PARAM_OFFSET2:.+]] = arith.constant 52 : i64
  // CHECK-DAG: %[[TARGET_OFFSET0:.+]] = arith.constant 100 : index
  // CHECK-DAG: %[[TARGET_OFFSET1:.+]] = arith.constant 101 : index
  // CHECK-DAG: %[[TARGET_OFFSET2:.+]] = arith.constant 102 : index
  // CHECK-DAG: %[[LENGTH0:.+]] = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH1:.+]] = arith.constant 201 : index
  // CHECK-DAG: %[[LENGTH2:.+]] = arith.constant 202 : index
  // CHECK-DAG: %[[TARGET_SIZE:.+]] = arith.constant 300 : index
  // CHECK:    = stream.cmd.parameter.gather await(%[[WAIT]]) => {
  // CHECK-NEXT:   %[[SCOPE]]::%[[KEY0]][%[[PARAM_OFFSET0]]] -> %[[TARGET]][%[[TARGET_OFFSET0]] for %[[LENGTH0]]] : !stream.resource<transient>{%[[TARGET_SIZE]]},
  // CHECK-NEXT:   %[[SCOPE]]::%[[KEY1]][%[[PARAM_OFFSET1]]] -> %[[TARGET]][%[[TARGET_OFFSET1]] for %[[LENGTH1]]] : !stream.resource<transient>{%[[TARGET_SIZE]]},
  // CHECK-NEXT:   %[[SCOPE]]::%[[KEY2]][%[[PARAM_OFFSET2]]] -> %[[TARGET]][%[[TARGET_OFFSET2]] for %[[LENGTH2]]] : !stream.resource<transient>{%[[TARGET_SIZE]]}
  // CHECK-NEXT: } => !stream.timepoint
  %timepoint = stream.cmd.parameter.gather await(%wait) => {
    %scope::%key0[%param_offset0] -> %target[%target_offset0 for %length0] : !stream.resource<transient>{%target_size},
    %scope::%key1[%param_offset1] -> %target[%target_offset1 for %length1] : !stream.resource<transient>{%target_size},
    %scope::%key2[%param_offset2] -> %target[%target_offset2 for %length2] : !stream.resource<transient>{%target_size}
  } => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}

// -----

// CHECK-LABEL: @parameterGatherNoScope
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[TARGET:.+]]: !stream.resource<transient>, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer)
util.func private @parameterGatherNoScope(%wait: !stream.timepoint, %target: !stream.resource<transient>, %key0: !util.buffer, %key1: !util.buffer) -> !stream.timepoint {
  %param_offset0 = arith.constant 50 : i64
  %param_offset1 = arith.constant 51 : i64
  %target_offset0 = arith.constant 100 : index
  %target_offset1 = arith.constant 101 : index
  %length0 = arith.constant 200 : index
  %length1 = arith.constant 201 : index
  %target_size = arith.constant 300 : index
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 50 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 51 : i64
  // CHECK-DAG: %[[TARGET_OFFSET0:.+]] = arith.constant 100 : index
  // CHECK-DAG: %[[TARGET_OFFSET1:.+]] = arith.constant 101 : index
  // CHECK-DAG: %[[LENGTH0:.+]] = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH1:.+]] = arith.constant 201 : index
  // CHECK-DAG: %[[TARGET_SIZE:.+]] = arith.constant 300 : index
  // CHECK:    = stream.cmd.parameter.gather await(%[[WAIT]]) => {
  // CHECK-NEXT:   %[[KEY0]][%[[PARAM_OFFSET0]]] -> %[[TARGET]][%[[TARGET_OFFSET0]] for %[[LENGTH0]]] : !stream.resource<transient>{%[[TARGET_SIZE]]},
  // CHECK-NEXT:   %[[KEY1]][%[[PARAM_OFFSET1]]] -> %[[TARGET]][%[[TARGET_OFFSET1]] for %[[LENGTH1]]] : !stream.resource<transient>{%[[TARGET_SIZE]]}
  // CHECK-NEXT: } => !stream.timepoint
  %timepoint = stream.cmd.parameter.gather await(%wait) => {
    %key0[%param_offset0] -> %target[%target_offset0 for %length0] : !stream.resource<transient>{%target_size},
    %key1[%param_offset1] -> %target[%target_offset1 for %length1] : !stream.resource<transient>{%target_size}
  } => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}

// -----

// CHECK-LABEL: @parameterScatter
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[SOURCE:.+]]: !stream.resource<transient>, %[[SCOPE:.+]]: !util.buffer, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer, %[[KEY2:.+]]: !util.buffer)
util.func private @parameterScatter(%wait: !stream.timepoint, %source: !stream.resource<transient>, %scope: !util.buffer, %key0: !util.buffer, %key1: !util.buffer, %key2: !util.buffer) -> !stream.timepoint {
  %param_offset0 = arith.constant 50 : i64
  %param_offset1 = arith.constant 51 : i64
  %param_offset2 = arith.constant 52 : i64
  %source_offset0 = arith.constant 100 : index
  %source_offset1 = arith.constant 101 : index
  %source_offset2 = arith.constant 102 : index
  %length0 = arith.constant 200 : index
  %length1 = arith.constant 201 : index
  %length2 = arith.constant 202 : index
  %source_size = arith.constant 300 : index
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 50 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 51 : i64
  // CHECK-DAG: %[[PARAM_OFFSET2:.+]] = arith.constant 52 : i64
  // CHECK-DAG: %[[SOURCE_OFFSET0:.+]] = arith.constant 100 : index
  // CHECK-DAG: %[[SOURCE_OFFSET1:.+]] = arith.constant 101 : index
  // CHECK-DAG: %[[SOURCE_OFFSET2:.+]] = arith.constant 102 : index
  // CHECK-DAG: %[[LENGTH0:.+]] = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH1:.+]] = arith.constant 201 : index
  // CHECK-DAG: %[[LENGTH2:.+]] = arith.constant 202 : index
  // CHECK-DAG: %[[SOURCE_SIZE:.+]] = arith.constant 300 : index
  //      CHECK: stream.cmd.parameter.scatter await(%[[WAIT]]) => {
  // CHECK-NEXT:   %[[SOURCE]][%[[SOURCE_OFFSET0]] for %[[LENGTH0]]] : !stream.resource<transient>{%[[SOURCE_SIZE]]} -> %[[SCOPE]]::%[[KEY0]][%[[PARAM_OFFSET0]]],
  // CHECK-NEXT:   %[[SOURCE]][%[[SOURCE_OFFSET1]] for %[[LENGTH1]]] : !stream.resource<transient>{%[[SOURCE_SIZE]]} -> %[[SCOPE]]::%[[KEY1]][%[[PARAM_OFFSET1]]],
  // CHECK-NEXT:   %[[SOURCE]][%[[SOURCE_OFFSET2]] for %[[LENGTH2]]] : !stream.resource<transient>{%[[SOURCE_SIZE]]} -> %[[SCOPE]]::%[[KEY2]][%[[PARAM_OFFSET2]]]
  // CHECK-NEXT: } => !stream.timepoint
  %timepoint = stream.cmd.parameter.scatter await(%wait) => {
    %source[%source_offset0 for %length0] : !stream.resource<transient>{%source_size} -> %scope::%key0[%param_offset0],
    %source[%source_offset1 for %length1] : !stream.resource<transient>{%source_size} -> %scope::%key1[%param_offset1],
    %source[%source_offset2 for %length2] : !stream.resource<transient>{%source_size} -> %scope::%key2[%param_offset2]
  } => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}
