// RUN: iree-opt --split-input-file --iree-hal-conversion --canonicalize %s | FileCheck %s

util.global private @device : !hal.device

// CHECK-LABEL: @parameterLoad
// CHECK-SAME: (%[[WAIT:.+]]: !hal.fence) -> (!hal.buffer, !hal.buffer, !hal.fence)
util.func public @parameterLoad(%wait: !stream.timepoint) -> (!stream.resource<constant>, !stream.resource<constant>, !stream.timepoint) {
  %c50_i64 = arith.constant 50 : i64
  %c51_i64 = arith.constant 51 : i64
  %c100 = arith.constant 100 : index
  %c101 = arith.constant 101 : index
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  // CHECK-DAG: %[[SIGNAL:.+]] = hal.fence.create device(%[[DEVICE]] : !hal.device)
  // CHECK-DAG: %[[MEMORY_TYPE:.+]] = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
  // CHECK-DAG: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<"{{.+}}Transfer{{.+}}Dispatch{{.+}}"> : i32
  // CHECK: %[[BUFFERS:.+]]:2 = io_parameters.load<%[[DEVICE]] : !hal.device> affinity(%[[AFFINITY]])
  // CHECK-SAME: wait(%[[WAIT]]) signal(%[[SIGNAL]])
  // CHECK-SAME: type(%[[MEMORY_TYPE]]) usage(%[[BUFFER_USAGE]])
  // CHECK-NEXT: "scope"::"key0"[%c50_i64] : !hal.buffer{%c100}
  // CHECK-NEXT: "scope"::"key1"[%c51_i64] : !hal.buffer{%c101}
  %results:2, %result_timepoint = stream.parameter.load on(#hal.device.affinity<@device>) await(%wait) => {
    "scope"::"key0"[%c50_i64] : !stream.resource<constant>{%c100},
    "scope"::"key1"[%c51_i64] : !stream.resource<constant>{%c101}
  } => !stream.timepoint
  // CHECK: return %[[BUFFERS]]#0, %[[BUFFERS]]#1, %[[SIGNAL]]
  util.return %results#0, %results#1, %result_timepoint : !stream.resource<constant>, !stream.resource<constant>, !stream.timepoint
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @parameterLoadNoScope
// CHECK-SAME: (%[[WAIT:.+]]: !hal.fence) -> (!hal.buffer, !hal.fence)
util.func public @parameterLoadNoScope(%wait: !stream.timepoint) -> (!stream.resource<constant>, !stream.timepoint) {
  %c50_i64 = arith.constant 50 : i64
  %c100 = arith.constant 100 : index
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  // CHECK-DAG: %[[SIGNAL:.+]] = hal.fence.create device(%[[DEVICE]] : !hal.device)
  // CHECK-DAG: %[[MEMORY_TYPE:.+]] = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
  // CHECK-DAG: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<"{{.+}}Transfer{{.+}}Dispatch{{.+}}"> : i32
  // CHECK: %[[BUFFER:.+]] = io_parameters.load<%[[DEVICE]] : !hal.device> affinity(%[[AFFINITY]])
  // CHECK-SAME: wait(%[[WAIT]]) signal(%[[SIGNAL]])
  // CHECK-SAME: type(%[[MEMORY_TYPE]]) usage(%[[BUFFER_USAGE]])
  // CHECK-NEXT: "key"[%c50_i64] : !hal.buffer{%c100}
  %result, %result_timepoint = stream.parameter.load on(#hal.device.affinity<@device>) await(%wait) => {
    "key"[%c50_i64] : !stream.resource<constant>{%c100}
  } => !stream.timepoint
  // CHECK: return %[[BUFFER]], %[[SIGNAL]]
  util.return %result, %result_timepoint : !stream.resource<constant>, !stream.timepoint
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @parameterRead
// CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[TARGET:.+]]: !hal.buffer) -> !hal.fence
util.func public @parameterRead(%wait: !stream.timepoint, %target: !stream.resource<transient>) -> !stream.timepoint {
  %c50_i64 = arith.constant 50 : i64
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  // CHECK-DAG: %[[SIGNAL:.+]] = hal.fence.create device(%[[DEVICE]] : !hal.device)
  // CHECK: io_parameters.gather<%[[DEVICE]] : !hal.device> affinity(%[[AFFINITY]])
  // CHECK-SAME: wait(%[[WAIT]]) signal(%[[SIGNAL]])
  // CHECK-NEXT:   "scope"::"key"[%c50_i64] -> %[[TARGET]][%c100 for %c200] : !hal.buffer
  %timepoint = stream.parameter.read on(#hal.device.affinity<@device>) await(%wait) => "scope"::"key"[%c50_i64] -> %target[%c100 for %c200] : !stream.resource<transient>{%c300} => !stream.timepoint
  // CHECK: return %[[SIGNAL]]
  util.return %timepoint : !stream.timepoint
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @parameterWrite
// CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[SOURCE:.+]]: !hal.buffer) -> !hal.fence
util.func public @parameterWrite(%wait: !stream.timepoint, %source: !stream.resource<transient>) -> !stream.timepoint {
  %c50_i64 = arith.constant 50 : i64
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  // CHECK-DAG: %[[SIGNAL:.+]] = hal.fence.create device(%[[DEVICE]] : !hal.device)
  // CHECK: io_parameters.scatter<%[[DEVICE]] : !hal.device> affinity(%[[AFFINITY]])
  // CHECK-SAME: wait(%[[WAIT]]) signal(%[[SIGNAL]])
  // CHECK-NEXT:   %[[SOURCE]][%c100 for %c200] : !hal.buffer -> "scope"::"key"[%c50_i64]
  %timepoint = stream.parameter.write on(#hal.device.affinity<@device>) await(%wait) => %source[%c100 for %c200] : !stream.resource<transient>{%c300} -> "scope"::"key"[%c50_i64] => !stream.timepoint
  // CHECK: return %[[SIGNAL]]
  util.return %timepoint : !stream.timepoint
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @parameterGather
// CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[TARGET:.+]]: !hal.buffer) -> !hal.fence
util.func public @parameterGather(%wait: !stream.timepoint, %target: !stream.resource<transient>) -> !stream.timepoint {
  %c50_i64 = arith.constant 50 : i64
  %c51_i64 = arith.constant 51 : i64
  %c52_i64 = arith.constant 52 : i64
  %c100 = arith.constant 100 : index
  %c101 = arith.constant 101 : index
  %c102 = arith.constant 102 : index
  %c200 = arith.constant 200 : index
  %c201 = arith.constant 201 : index
  %c202 = arith.constant 202 : index
  %c300 = arith.constant 300 : index
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  // CHECK-DAG: %[[SIGNAL:.+]] = hal.fence.create device(%[[DEVICE]] : !hal.device)
  // CHECK: io_parameters.gather<%[[DEVICE]] : !hal.device> affinity(%[[AFFINITY]])
  // CHECK-SAME: wait(%[[WAIT]]) signal(%[[SIGNAL]])
  // CHECK-NEXT:   "scope"::"key0"[%c50_i64] -> %[[TARGET]][%c100 for %c200] : !hal.buffer,
  // CHECK-NEXT:   "scope"::"key1"[%c51_i64] -> %[[TARGET]][%c101 for %c201] : !hal.buffer,
  // CHECK-NEXT:   "scope"::"key2"[%c52_i64] -> %[[TARGET]][%c102 for %c202] : !hal.buffer
  %timepoint = stream.parameter.gather on(#hal.device.affinity<@device>) await(%wait) => {
    "scope"::"key0"[%c50_i64] -> %target[%c100 for %c200] : !stream.resource<transient>{%c300},
    "scope"::"key1"[%c51_i64] -> %target[%c101 for %c201] : !stream.resource<transient>{%c300},
    "scope"::"key2"[%c52_i64] -> %target[%c102 for %c202] : !stream.resource<transient>{%c300}
  } => !stream.timepoint
  // CHECK: return %[[SIGNAL]]
  util.return %timepoint : !stream.timepoint
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @parameterGatherNoScope
// CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[TARGET:.+]]: !hal.buffer) -> !hal.fence
util.func public @parameterGatherNoScope(%wait: !stream.timepoint, %target: !stream.resource<transient>) -> !stream.timepoint {
  %c50_i64 = arith.constant 50 : i64
  %c51_i64 = arith.constant 51 : i64
  %c100 = arith.constant 100 : index
  %c101 = arith.constant 101 : index
  %c200 = arith.constant 200 : index
  %c201 = arith.constant 201 : index
  %c300 = arith.constant 300 : index
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  // CHECK-DAG: %[[SIGNAL:.+]] = hal.fence.create device(%[[DEVICE]] : !hal.device)
  // CHECK: io_parameters.gather<%[[DEVICE]] : !hal.device> affinity(%[[AFFINITY]])
  // CHECK-SAME: wait(%[[WAIT]]) signal(%[[SIGNAL]])
  // CHECK-NEXT:   "key0"[%c50_i64] -> %[[TARGET]][%c100 for %c200] : !hal.buffer,
  // CHECK-NEXT:   "key1"[%c51_i64] -> %[[TARGET]][%c101 for %c201] : !hal.buffer
  %timepoint = stream.parameter.gather on(#hal.device.affinity<@device>) await(%wait) => {
    "key0"[%c50_i64] -> %target[%c100 for %c200] : !stream.resource<transient>{%c300},
    "key1"[%c51_i64] -> %target[%c101 for %c201] : !stream.resource<transient>{%c300}
  } => !stream.timepoint
  // CHECK: return %[[SIGNAL]]
  util.return %timepoint : !stream.timepoint
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @parameterScatter
// CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[SOURCE:.+]]: !hal.buffer) -> !hal.fence
util.func public @parameterScatter(%wait: !stream.timepoint, %source: !stream.resource<transient>) -> !stream.timepoint {
  %c50_i64 = arith.constant 50 : i64
  %c51_i64 = arith.constant 51 : i64
  %c52_i64 = arith.constant 52 : i64
  %c100 = arith.constant 100 : index
  %c101 = arith.constant 101 : index
  %c102 = arith.constant 102 : index
  %c200 = arith.constant 200 : index
  %c201 = arith.constant 201 : index
  %c202 = arith.constant 202 : index
  %c300 = arith.constant 300 : index
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  // CHECK-DAG: %[[SIGNAL:.+]] = hal.fence.create device(%[[DEVICE]] : !hal.device)
  // CHECK: io_parameters.scatter<%[[DEVICE]] : !hal.device> affinity(%[[AFFINITY]])
  // CHECK-SAME: wait(%[[WAIT]]) signal(%[[SIGNAL]])
  // CHECK-NEXT:   %[[SOURCE]][%c100 for %c200] : !hal.buffer -> "scope"::"key0"[%c50_i64],
  // CHECK-NEXT:   %[[SOURCE]][%c101 for %c201] : !hal.buffer -> "scope"::"key1"[%c51_i64],
  // CHECK-NEXT:   %[[SOURCE]][%c102 for %c202] : !hal.buffer -> "scope"::"key2"[%c52_i64]
  // CHECK-NEXT: }
  %timepoint = stream.parameter.scatter on(#hal.device.affinity<@device>) await(%wait) => {
    %source[%c100 for %c200] : !stream.resource<transient>{%c300} -> "scope"::"key0"[%c50_i64],
    %source[%c101 for %c201] : !stream.resource<transient>{%c300} -> "scope"::"key1"[%c51_i64],
    %source[%c102 for %c202] : !stream.resource<transient>{%c300} -> "scope"::"key2"[%c52_i64]
  } => !stream.timepoint
  // CHECK: return %[[SIGNAL]]
  util.return %timepoint : !stream.timepoint
}
