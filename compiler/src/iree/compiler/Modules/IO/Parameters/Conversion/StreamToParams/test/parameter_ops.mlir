// RUN: iree-opt --split-input-file --iree-hal-conversion --canonicalize %s | FileCheck %s

util.global private @device : !hal.device

// CHECK-LABEL: @parameterLoad
// CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[SCOPE:.+]]: !util.buffer, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer) -> (!hal.buffer, !hal.buffer, !hal.fence)
util.func public @parameterLoad(%wait: !stream.timepoint, %scope: !util.buffer, %key0: !util.buffer, %key1: !util.buffer) -> (!stream.resource<constant>, !stream.resource<constant>, !stream.timepoint) {
  %param_offset0 = arith.constant 50 : i64
  %param_offset1 = arith.constant 51 : i64
  %size0 = arith.constant 100 : index
  %size1 = arith.constant 101 : index
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  // CHECK-DAG: %[[SIGNAL:.+]] = hal.fence.create device(%[[DEVICE]] : !hal.device)
  // CHECK-DAG: %[[MEMORY_TYPE:.+]], %[[BUFFER_USAGE:.+]] = hal.allocator.resolve_memory_properties for(#hal.device.affinity<@device>) lifetime(constant)
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 50 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 51 : i64
  // CHECK-DAG: %[[SIZE0:.+]] = arith.constant 100 : index
  // CHECK-DAG: %[[SIZE1:.+]] = arith.constant 101 : index
  // CHECK: %[[BUFFERS:.+]]:2 = io_parameters.load<%[[DEVICE]] : !hal.device> affinity(%[[AFFINITY]])
  // CHECK-SAME: wait(%[[WAIT]]) signal(%[[SIGNAL]])
  // CHECK-SAME: type(%[[MEMORY_TYPE]]) usage(%[[BUFFER_USAGE]])
  // CHECK-NEXT: %[[SCOPE]]::%[[KEY0]][%[[PARAM_OFFSET0]]] : !hal.buffer{%[[SIZE0]]}
  // CHECK-NEXT: %[[SCOPE]]::%[[KEY1]][%[[PARAM_OFFSET1]]] : !hal.buffer{%[[SIZE1]]}
  %results:2, %result_timepoint = stream.cmd.parameter.load on(#hal.device.affinity<@device>) await(%wait) => {
    %scope::%key0[%param_offset0] : !stream.resource<constant>{%size0},
    %scope::%key1[%param_offset1] : !stream.resource<constant>{%size1}
  } => !stream.timepoint
  // CHECK: return %[[BUFFERS]]#0, %[[BUFFERS]]#1, %[[SIGNAL]]
  util.return %results#0, %results#1, %result_timepoint : !stream.resource<constant>, !stream.resource<constant>, !stream.timepoint
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @parameterLoadNoScope
// CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[KEY:.+]]: !util.buffer) -> (!hal.buffer, !hal.fence)
util.func public @parameterLoadNoScope(%wait: !stream.timepoint, %key: !util.buffer) -> (!stream.resource<constant>, !stream.timepoint) {
  %param_offset = arith.constant 50 : i64
  %size = arith.constant 100 : index
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  // CHECK-DAG: %[[SIGNAL:.+]] = hal.fence.create device(%[[DEVICE]] : !hal.device)
  // CHECK-DAG: %[[MEMORY_TYPE:.+]], %[[BUFFER_USAGE:.+]] = hal.allocator.resolve_memory_properties for(#hal.device.affinity<@device>) lifetime(constant)
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 50 : i64
  // CHECK-DAG: %[[SIZE:.+]] = arith.constant 100 : index
  // CHECK: %[[BUFFER:.+]] = io_parameters.load<%[[DEVICE]] : !hal.device> affinity(%[[AFFINITY]])
  // CHECK-SAME: wait(%[[WAIT]]) signal(%[[SIGNAL]])
  // CHECK-SAME: type(%[[MEMORY_TYPE]]) usage(%[[BUFFER_USAGE]])
  // CHECK-NEXT: %[[KEY]][%[[PARAM_OFFSET]]] : !hal.buffer{%[[SIZE]]}
  %result, %result_timepoint = stream.cmd.parameter.load on(#hal.device.affinity<@device>) await(%wait) => {
    %key[%param_offset] : !stream.resource<constant>{%size}
  } => !stream.timepoint
  // CHECK: return %[[BUFFER]], %[[SIGNAL]]
  util.return %result, %result_timepoint : !stream.resource<constant>, !stream.timepoint
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @parameterRead
// CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[TARGET:.+]]: !hal.buffer, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer) -> !hal.fence
util.func public @parameterRead(%wait: !stream.timepoint, %target: !stream.resource<transient>, %scope: !util.buffer, %key: !util.buffer) -> !stream.timepoint {
  %param_offset = arith.constant 50 : i64
  %target_offset = arith.constant 100 : index
  %length = arith.constant 200 : index
  %target_size = arith.constant 300 : index
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  // CHECK-DAG: %[[SIGNAL:.+]] = hal.fence.create device(%[[DEVICE]] : !hal.device)
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 50 : i64
  // CHECK-DAG: %[[TARGET_OFFSET:.+]] = arith.constant 100 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 200 : index
  // CHECK: io_parameters.gather<%[[DEVICE]] : !hal.device> affinity(%[[AFFINITY]])
  // CHECK-SAME: wait(%[[WAIT]]) signal(%[[SIGNAL]])
  // CHECK-NEXT:   %[[SCOPE]]::%[[KEY]][%[[PARAM_OFFSET]]] -> %[[TARGET]][%[[TARGET_OFFSET]] for %[[LENGTH]]] : !hal.buffer
  %timepoint = stream.cmd.parameter.read on(#hal.device.affinity<@device>) await(%wait) => %scope::%key[%param_offset] -> %target[%target_offset for %length] : !stream.resource<transient>{%target_size} => !stream.timepoint
  // CHECK: return %[[SIGNAL]]
  util.return %timepoint : !stream.timepoint
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @parameterWrite
// CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[SOURCE:.+]]: !hal.buffer, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer) -> !hal.fence
util.func public @parameterWrite(%wait: !stream.timepoint, %source: !stream.resource<transient>, %scope: !util.buffer, %key: !util.buffer) -> !stream.timepoint {
  %param_offset = arith.constant 50 : i64
  %source_offset = arith.constant 100 : index
  %length = arith.constant 200 : index
  %source_size = arith.constant 300 : index
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  // CHECK-DAG: %[[SIGNAL:.+]] = hal.fence.create device(%[[DEVICE]] : !hal.device)
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 50 : i64
  // CHECK-DAG: %[[SOURCE_OFFSET:.+]] = arith.constant 100 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 200 : index
  // CHECK: io_parameters.scatter<%[[DEVICE]] : !hal.device> affinity(%[[AFFINITY]])
  // CHECK-SAME: wait(%[[WAIT]]) signal(%[[SIGNAL]])
  // CHECK-NEXT:   %[[SOURCE]][%[[SOURCE_OFFSET]] for %[[LENGTH]]] : !hal.buffer -> %[[SCOPE]]::%[[KEY]][%[[PARAM_OFFSET]]]
  %timepoint = stream.cmd.parameter.write on(#hal.device.affinity<@device>) await(%wait) => %source[%source_offset for %length] : !stream.resource<transient>{%source_size} -> %scope::%key[%param_offset] => !stream.timepoint
  // CHECK: return %[[SIGNAL]]
  util.return %timepoint : !stream.timepoint
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @parameterGather
// CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[TARGET:.+]]: !hal.buffer, %[[SCOPE:.+]]: !util.buffer, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer, %[[KEY2:.+]]: !util.buffer) -> !hal.fence
util.func public @parameterGather(%wait: !stream.timepoint, %target: !stream.resource<transient>, %scope: !util.buffer, %key0: !util.buffer, %key1: !util.buffer, %key2: !util.buffer) -> !stream.timepoint {
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
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  // CHECK-DAG: %[[SIGNAL:.+]] = hal.fence.create device(%[[DEVICE]] : !hal.device)
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 50 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 51 : i64
  // CHECK-DAG: %[[PARAM_OFFSET2:.+]] = arith.constant 52 : i64
  // CHECK-DAG: %[[TARGET_OFFSET0:.+]] = arith.constant 100 : index
  // CHECK-DAG: %[[TARGET_OFFSET1:.+]] = arith.constant 101 : index
  // CHECK-DAG: %[[TARGET_OFFSET2:.+]] = arith.constant 102 : index
  // CHECK-DAG: %[[LENGTH0:.+]] = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH1:.+]] = arith.constant 201 : index
  // CHECK-DAG: %[[LENGTH2:.+]] = arith.constant 202 : index
  // CHECK: io_parameters.gather<%[[DEVICE]] : !hal.device> affinity(%[[AFFINITY]])
  // CHECK-SAME: wait(%[[WAIT]]) signal(%[[SIGNAL]])
  // CHECK-NEXT:   %[[SCOPE]]::%[[KEY0]][%[[PARAM_OFFSET0]]] -> %[[TARGET]][%[[TARGET_OFFSET0]] for %[[LENGTH0]]] : !hal.buffer,
  // CHECK-NEXT:   %[[SCOPE]]::%[[KEY1]][%[[PARAM_OFFSET1]]] -> %[[TARGET]][%[[TARGET_OFFSET1]] for %[[LENGTH1]]] : !hal.buffer,
  // CHECK-NEXT:   %[[SCOPE]]::%[[KEY2]][%[[PARAM_OFFSET2]]] -> %[[TARGET]][%[[TARGET_OFFSET2]] for %[[LENGTH2]]] : !hal.buffer
  %timepoint = stream.cmd.parameter.gather on(#hal.device.affinity<@device>) await(%wait) => {
    %scope::%key0[%param_offset0] -> %target[%target_offset0 for %length0] : !stream.resource<transient>{%target_size},
    %scope::%key1[%param_offset1] -> %target[%target_offset1 for %length1] : !stream.resource<transient>{%target_size},
    %scope::%key2[%param_offset2] -> %target[%target_offset2 for %length2] : !stream.resource<transient>{%target_size}
  } => !stream.timepoint
  // CHECK: return %[[SIGNAL]]
  util.return %timepoint : !stream.timepoint
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @parameterGatherNoScope
// CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[TARGET:.+]]: !hal.buffer, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer) -> !hal.fence
util.func public @parameterGatherNoScope(%wait: !stream.timepoint, %target: !stream.resource<transient>, %key0: !util.buffer, %key1: !util.buffer) -> !stream.timepoint {
  %param_offset0 = arith.constant 50 : i64
  %param_offset1 = arith.constant 51 : i64
  %target_offset0 = arith.constant 100 : index
  %target_offset1 = arith.constant 101 : index
  %length0 = arith.constant 200 : index
  %length1 = arith.constant 201 : index
  %target_size = arith.constant 300 : index
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  // CHECK-DAG: %[[SIGNAL:.+]] = hal.fence.create device(%[[DEVICE]] : !hal.device)
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 50 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 51 : i64
  // CHECK-DAG: %[[TARGET_OFFSET0:.+]] = arith.constant 100 : index
  // CHECK-DAG: %[[TARGET_OFFSET1:.+]] = arith.constant 101 : index
  // CHECK-DAG: %[[LENGTH0:.+]] = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH1:.+]] = arith.constant 201 : index
  // CHECK: io_parameters.gather<%[[DEVICE]] : !hal.device> affinity(%[[AFFINITY]])
  // CHECK-SAME: wait(%[[WAIT]]) signal(%[[SIGNAL]])
  // CHECK-NEXT:   %[[KEY0]][%[[PARAM_OFFSET0]]] -> %[[TARGET]][%[[TARGET_OFFSET0]] for %[[LENGTH0]]] : !hal.buffer,
  // CHECK-NEXT:   %[[KEY1]][%[[PARAM_OFFSET1]]] -> %[[TARGET]][%[[TARGET_OFFSET1]] for %[[LENGTH1]]] : !hal.buffer
  %timepoint = stream.cmd.parameter.gather on(#hal.device.affinity<@device>) await(%wait) => {
    %key0[%param_offset0] -> %target[%target_offset0 for %length0] : !stream.resource<transient>{%target_size},
    %key1[%param_offset1] -> %target[%target_offset1 for %length1] : !stream.resource<transient>{%target_size}
  } => !stream.timepoint
  // CHECK: return %[[SIGNAL]]
  util.return %timepoint : !stream.timepoint
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @parameterScatter
// CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[SOURCE:.+]]: !hal.buffer, %[[SCOPE:.+]]: !util.buffer, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer, %[[KEY2:.+]]: !util.buffer) -> !hal.fence
util.func public @parameterScatter(%wait: !stream.timepoint, %source: !stream.resource<transient>, %scope: !util.buffer, %key0: !util.buffer, %key1: !util.buffer, %key2: !util.buffer) -> !stream.timepoint {
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
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  // CHECK-DAG: %[[SIGNAL:.+]] = hal.fence.create device(%[[DEVICE]] : !hal.device)
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 50 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 51 : i64
  // CHECK-DAG: %[[PARAM_OFFSET2:.+]] = arith.constant 52 : i64
  // CHECK-DAG: %[[SOURCE_OFFSET0:.+]] = arith.constant 100 : index
  // CHECK-DAG: %[[SOURCE_OFFSET1:.+]] = arith.constant 101 : index
  // CHECK-DAG: %[[SOURCE_OFFSET2:.+]] = arith.constant 102 : index
  // CHECK-DAG: %[[LENGTH0:.+]] = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH1:.+]] = arith.constant 201 : index
  // CHECK-DAG: %[[LENGTH2:.+]] = arith.constant 202 : index
  // CHECK: io_parameters.scatter<%[[DEVICE]] : !hal.device> affinity(%[[AFFINITY]])
  // CHECK-SAME: wait(%[[WAIT]]) signal(%[[SIGNAL]])
  // CHECK-NEXT:   %[[SOURCE]][%[[SOURCE_OFFSET0]] for %[[LENGTH0]]] : !hal.buffer -> %[[SCOPE]]::%[[KEY0]][%[[PARAM_OFFSET0]]],
  // CHECK-NEXT:   %[[SOURCE]][%[[SOURCE_OFFSET1]] for %[[LENGTH1]]] : !hal.buffer -> %[[SCOPE]]::%[[KEY1]][%[[PARAM_OFFSET1]]],
  // CHECK-NEXT:   %[[SOURCE]][%[[SOURCE_OFFSET2]] for %[[LENGTH2]]] : !hal.buffer -> %[[SCOPE]]::%[[KEY2]][%[[PARAM_OFFSET2]]]
  // CHECK-NEXT: }
  %timepoint = stream.cmd.parameter.scatter on(#hal.device.affinity<@device>) await(%wait) => {
    %source[%source_offset0 for %length0] : !stream.resource<transient>{%source_size} -> %scope::%key0[%param_offset0],
    %source[%source_offset1 for %length1] : !stream.resource<transient>{%source_size} -> %scope::%key1[%param_offset1],
    %source[%source_offset2 for %length2] : !stream.resource<transient>{%source_size} -> %scope::%key2[%param_offset2]
  } => !stream.timepoint
  // CHECK: return %[[SIGNAL]]
  util.return %timepoint : !stream.timepoint
}
