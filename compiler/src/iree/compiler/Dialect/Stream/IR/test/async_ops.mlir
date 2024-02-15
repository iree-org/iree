// RUN: iree-opt --split-input-file %s --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @asyncAlloca
util.func private @asyncAlloca(%arg0: index) -> !stream.resource<transient> {
  // CHECK: = stream.async.alloca : !stream.resource<transient>{%arg0}
  %0 = stream.async.alloca : !stream.resource<transient>{%arg0}
  util.return %0 : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @asyncConstant
util.func private @asyncConstant(%arg0: index) -> !stream.resource<transient> {
  // CHECK: = stream.async.constant : !stream.resource<transient>{%arg0} = dense<3> : tensor<8xi32>
  %0 = stream.async.constant : !stream.resource<transient>{%arg0} = dense<3> : tensor<8xi32>
  util.return %0 : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @asyncSplat
util.func private @asyncSplat(%arg0: index, %arg1: i32) -> !stream.resource<*> {
  // CHECK: = stream.async.splat %arg1 : i32 -> !stream.resource<*>{%arg0}
  %0 = stream.async.splat %arg1 : i32 -> !stream.resource<*>{%arg0}
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncClone
util.func private @asyncClone(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  // CHECK: = stream.async.clone %arg0 : !stream.resource<*>{%arg1} -> !stream.resource<*>{%arg1}
  %0 = stream.async.clone %arg0 : !stream.resource<*>{%arg1} -> !stream.resource<*>{%arg1}
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncSlice
util.func private @asyncSlice(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: = stream.async.slice %arg0[%c0 to %c128] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c128}
  %0 = stream.async.slice %arg0[%c0 to %c128] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c128}
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncFill
util.func private @asyncFill(%arg0: !stream.resource<*>, %arg1: index, %arg2: i32) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: = stream.async.fill %arg2, %arg0[%c0 to %c128 for %c128] : i32 -> %arg0 as !stream.resource<*>{%arg1}
  %0 = stream.async.fill %arg2, %arg0[%c0 to %c128 for %c128] : i32 -> %arg0 as !stream.resource<*>{%arg1}
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncUpdate
util.func private @asyncUpdate(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: = stream.async.update %arg2, %arg0[%c0 to %c128] : !stream.resource<*>{%arg3} -> %arg0 as !stream.resource<*>{%arg1}
  %0 = stream.async.update %arg2, %arg0[%c0 to %c128] : !stream.resource<*>{%arg3} -> %arg0 as !stream.resource<*>{%arg1}
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncCopy
util.func private @asyncCopy(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: = stream.async.copy %arg2[%c0 to %c128], %arg0[%c0 to %c128], %c128 : !stream.resource<*>{%arg3} -> %arg0 as !stream.resource<*>{%arg1}
  %0 = stream.async.copy %arg2[%c0 to %c128], %arg0[%c0 to %c128], %c128 : !stream.resource<*>{%arg3} -> %arg0 as !stream.resource<*>{%arg1}
  util.return %0 : !stream.resource<*>
}

// -----

// This covers all_gather, all_reduce, and reduce_scatter variants.

// CHECK-LABEL: @asyncCollectiveAllGather
util.func private @asyncCollectiveAllGather(
    // CHECK-SAME: %[[CHANNEL:.+]]: !stream.channel,
    %channel: !stream.channel,
    // CHECK-SAME: %[[SEND:[a-z0-9]+]]: !stream.resource<*>, %[[SEND_SIZE:[a-z0-9]+]]: index,
    %send: !stream.resource<*>, %send_size: index,
    // CHECK-SAME: %[[RECV_SIZE:[a-z0-9]+]]: index, %[[COUNT:[a-z0-9]+]]: index)
    %recv_size: index, %count: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[RECV:.+]] = stream.async.alloca
  %recv = stream.async.alloca : !stream.resource<*>{%recv_size}
  // CHECK: = stream.async.collective<all_gather : f32>[%[[COUNT]]]
  %0 = stream.async.collective<all_gather : f32>[%count]
      // CHECK-SAME: on(#hal.affinity.queue<[0]>) channel(%[[CHANNEL]])
      on(#hal.affinity.queue<[0]>) channel(%channel)
      // CHECK-SAME: %[[SEND]][%c0 to %[[SEND_SIZE]] for %[[SEND_SIZE]]],
      %send[%c0 to %send_size for %send_size],
      // CHECK-SAME: %[[RECV]][%c0 to %[[RECV_SIZE]] for %[[RECV_SIZE]]] :
      %recv[%c0 to %recv_size for %recv_size] :
      // CHECK-SAME: !stream.resource<*>{%[[SEND_SIZE]]} -> %[[RECV]] as !stream.resource<*>{%[[RECV_SIZE]]}
      !stream.resource<*>{%send_size} -> %recv as !stream.resource<*>{%recv_size}
  util.return %0 : !stream.resource<*>
}

// -----

// This covers broadcast and reduce variants.

// CHECK-LABEL: @asyncCollectiveBroadcast
util.func private @asyncCollectiveBroadcast(
    // CHECK-SAME: %[[CHANNEL:.+]]: !stream.channel,
    %channel: !stream.channel,
    // CHECK-SAME: %[[RANK:[a-z0-9]+]]: i32,
    %rank: i32,
    // CHECK-SAME: %[[SEND:[a-z0-9]+]]: !stream.resource<*>, %[[SEND_SIZE:[a-z0-9]+]]: index,
    %send: !stream.resource<*>, %send_size: index,
    // CHECK-SAME: %[[RECV_SIZE:[a-z0-9]+]]: index, %[[COUNT:[a-z0-9]+]]: index)
    %recv_size: index, %count: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[RECV:.+]] = stream.async.alloca
  %recv = stream.async.alloca : !stream.resource<*>{%recv_size}
  // CHECK: = stream.async.collective<broadcast : f32>[%[[COUNT]]]
  %0 = stream.async.collective<broadcast : f32>[%count]
      // CHECK-SAME: on(#hal.affinity.queue<[0]>) channel(%[[CHANNEL]]) source(%[[RANK]])
      on(#hal.affinity.queue<[0]>) channel(%channel) source(%rank)
      // CHECK-SAME: %[[SEND]][%c0 to %[[SEND_SIZE]] for %[[SEND_SIZE]]],
      %send[%c0 to %send_size for %send_size],
      // CHECK-SAME: %[[RECV]][%c0 to %[[RECV_SIZE]] for %[[RECV_SIZE]]] :
      %recv[%c0 to %recv_size for %recv_size] :
      // CHECK-SAME: !stream.resource<*>{%[[SEND_SIZE]]} -> %[[RECV]] as !stream.resource<*>{%[[RECV_SIZE]]}
      !stream.resource<*>{%send_size} -> %recv as !stream.resource<*>{%recv_size}
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncTransfer
util.func private @asyncTransfer(%arg0: !stream.resource<constant>, %arg1: index) -> !stream.resource<staging> {
  // CHECK: = stream.async.transfer %arg0 : !stream.resource<constant>{%arg1} -> !stream.resource<staging>{%arg1}
  %0 = stream.async.transfer %arg0 : !stream.resource<constant>{%arg1} -> !stream.resource<staging>{%arg1}
  util.return %0 : !stream.resource<staging>
}

// -----

// CHECK-LABEL: @asyncLoad
util.func private @asyncLoad(%arg0: !stream.resource<staging>, %arg1: index) -> f32 {
  %c0 = arith.constant 0 : index
  // CHECK: = stream.async.load %arg0[%c0] : !stream.resource<staging>{%arg1} -> f32
  %0 = stream.async.load %arg0[%c0] : !stream.resource<staging>{%arg1} -> f32
  util.return %0 : f32
}

// -----

// CHECK-LABEL: @asyncStore
util.func private @asyncStore(%arg0: !stream.resource<staging>, %arg1: index, %arg2: f32) -> !stream.resource<staging> {
  %c0 = arith.constant 0 : index
  // CHECK: = stream.async.store %arg2, %arg0[%c0] : f32 -> %arg0 as !stream.resource<staging>{%arg1}
  %0 = stream.async.store %arg2, %arg0[%c0] : f32 -> %arg0 as !stream.resource<staging>{%arg1}
  util.return %0 : !stream.resource<staging>
}

// -----

// CHECK-LABEL: @asyncDispatch
util.func private @asyncDispatch(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  // CHECK: = stream.async.dispatch @executable::@dispatch[%c1, %c2, %c3](%arg0[%c0 to %arg1 for %arg1], %c4) : (!stream.resource<*>{%arg1}, index) -> %arg0{%arg1}
  %0 = stream.async.dispatch @executable::@dispatch[%c1, %c2, %c3](%arg0[%c0 to %arg1 for %arg1], %c4) : (!stream.resource<*>{%arg1}, index) -> %arg0{%arg1}
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncDispatchNoInputs
util.func private @asyncDispatchNoInputs(%arg0: index) -> !stream.resource<*> {
  %c1 = arith.constant 1 : index
  // CHECK: = stream.async.dispatch @executable::@dispatch[%c1]() : () -> !stream.resource<*>{%arg0}
  %0 = stream.async.dispatch @executable::@dispatch[%c1]() : () -> !stream.resource<*>{%arg0}
  util.return %0 : !stream.resource<*>
}

// -----

stream.executable private @executable {
  stream.executable.export public @dispatch workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
    stream.return %arg0, %arg1, %arg0 : index, index, index
  }
  builtin.module {
    util.func private @dispatch() {
      util.return
    }
  }
}

// CHECK-LABEL: @asyncDispatchWithWorkgroupCount
util.func private @asyncDispatchWithWorkgroupCount(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  // CHECK: = stream.async.dispatch @executable::@dispatch[%c1, %c2](%arg0[%c0 to %arg1 for %arg1], %c4) : (!stream.resource<*>{%arg1}, index) -> %arg0{%arg1}
  %0 = stream.async.dispatch @executable::@dispatch[%c1, %c2](%arg0[%c0 to %arg1 for %arg1], %c4) : (!stream.resource<*>{%arg1}, index) -> %arg0{%arg1}
  util.return %0 : !stream.resource<*>
}

// -----

stream.executable private @executable {
  stream.executable.export public @dispatch workgroups(%arg0: index) -> (index, index, index) {
    stream.return %arg0, %arg0, %arg0 : index, index, index
  }
  builtin.module {
    util.func private @dispatch() {
      util.return
    }
  }
}

util.func private @asyncDispatchWithInvalidWorkload(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  // expected-error @+1 {{op workload mismatch; entry point expects 1 arguments but dispatch provides 2}}
  %0 = stream.async.dispatch @executable::@dispatch[%c1, %c2](%arg0[%c0 to %arg1 for %arg1], %c4) : (!stream.resource<*>{%arg1}, index) -> %arg0{%arg1}
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncDispatchNoWorkload
util.func private @asyncDispatchNoWorkload(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  // CHECK: = stream.async.dispatch @executable::@dispatch(%arg0[%c0 to %arg1 for %arg1], %c4) : (!stream.resource<*>{%arg1}, index) -> %arg0{%arg1}
  %0 = stream.async.dispatch @executable::@dispatch(%arg0[%c0 to %arg1 for %arg1], %c4) : (!stream.resource<*>{%arg1}, index) -> %arg0{%arg1}
  util.return %0 : !stream.resource<*>
}

// -----

stream.async.func private @asyncExtern(%arg0: !stream.resource<*>, %arg1: index) -> %arg0

// CHECK-LABEL: @asyncCall
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[SIZE0:.+]]: index)
util.func private @asyncCall(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK: = stream.async.call @asyncExtern(%[[ARG0]][%c0 to %[[SIZE0]] for %[[SIZE0]]], %[[SIZE0]]) : (!stream.resource<*>{%[[SIZE0]]}, index) -> %[[ARG0]]{%[[SIZE0]]}
  %call = stream.async.call @asyncExtern(%arg0[%c0 to %arg1 for %arg1], %arg1) : (!stream.resource<*>{%arg1}, index) -> %arg0{%arg1}
  util.return %call : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncExecute
util.func private @asyncExecute(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.timepoint) -> (!stream.resource<*>, !stream.timepoint) {
  // CHECK: = stream.async.execute await(%arg2) => with(%arg0 as %arg3: !stream.resource<*>{%arg1}) -> %arg0{%arg1} {
  %0:2 = stream.async.execute await(%arg2) => with(%arg0 as %arg3: !stream.resource<*>{%arg1}) -> %arg0 as !stream.resource<*>{%arg1} {
    // CHECK: %[[W:.+]] = stream.async.concurrent with(%arg3 as %arg4: !stream.resource<*>{%arg1}) -> %arg3{%arg1} {
    %1 = stream.async.concurrent with(%arg3 as %arg4: !stream.resource<*>{%arg1}) -> %arg3 as !stream.resource<*>{%arg1} {
      // CHECK: stream.yield %arg4 : !stream.resource<*>{%arg1}
      stream.yield %arg4 : !stream.resource<*>{%arg1}
    }
    // CHECK: stream.yield %[[W]] : !stream.resource<*>{%arg1}
    stream.yield %1 : !stream.resource<*>{%arg1}
  } => !stream.timepoint
  util.return %0#0, %0#1 : !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @asyncExecuteNoCaptures
util.func private @asyncExecuteNoCaptures(%arg0: index, %arg1: i32) -> (!stream.resource<*>, !stream.timepoint) {
  // CHECK: = stream.async.execute with() -> !stream.resource<*>{%arg0} {
  %0:2 = stream.async.execute with() -> !stream.resource<*>{%arg0} {
    // CHECK: %[[T:.+]] = stream.async.splat
    %1 = stream.async.splat %arg1 : i32 -> !stream.resource<*>{%arg0}
    // CHECK: stream.yield %[[T]] : !stream.resource<*>{%arg0}
    stream.yield %1 : !stream.resource<*>{%arg0}
  } => !stream.timepoint
  util.return %0#0, %0#1 : !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @asyncExecuteNoResults
util.func private @asyncExecuteNoResults(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.timepoint) -> (!stream.timepoint) {
  // CHECK: = stream.async.execute await(%arg2) => with(%arg0 as %arg3: !stream.resource<*>{%arg1}) {
  %0:1 = stream.async.execute await(%arg2) => with(%arg0 as %arg3: !stream.resource<*>{%arg1}) {
    // CHECK: stream.yield
    stream.yield
  } => !stream.timepoint
  util.return %0#0 : !stream.timepoint
}
