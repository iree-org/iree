// RUN: iree-opt --split-input-file %s --verify-diagnostics | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @asyncParameterLoad
// CHECK-SAME: (%[[SIZE:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func private @asyncParameterLoad(%size: index, %scope: !util.buffer, %key: !util.buffer) -> !stream.resource<constant> {
  // CHECK: %[[OFFSET:.+]] = arith.constant 128
  %offset = arith.constant 128 : i64
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.load %[[SCOPE]]::%[[KEY]][%[[OFFSET]]] : !stream.resource<constant>{%[[SIZE]]} => !stream.timepoint
  %result, %result_ready = stream.async.parameter.load %scope::%key[%offset] : !stream.resource<constant>{%size} => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<constant>{%[[SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<constant>{%size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @asyncParameterLoadNoScope
// CHECK-SAME: (%[[SIZE:.+]]: index, %[[KEY:.+]]: !util.buffer)
util.func private @asyncParameterLoadNoScope(%size: index, %key: !util.buffer) -> !stream.resource<constant> {
  // CHECK: %[[OFFSET:.+]] = arith.constant 96
  %offset = arith.constant 96 : i64
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.load %[[KEY]][%[[OFFSET]]] : !stream.resource<constant>{%[[SIZE]]} => !stream.timepoint
  %result, %result_ready = stream.async.parameter.load %key[%offset] : !stream.resource<constant>{%size} => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<constant>{%[[SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<constant>{%size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @asyncParameterLoadWithAffinity
// CHECK-SAME: (%[[SIZE:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func private @asyncParameterLoadWithAffinity(%size: index, %scope: !util.buffer, %key: !util.buffer) -> !stream.resource<constant> {
  // CHECK: %[[OFFSET:.+]] = arith.constant 100000000
  %offset = arith.constant 100000000 : i64
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.load on(#hal.device.affinity<@device>) %[[SCOPE]]::%[[KEY]][%[[OFFSET]]] : !stream.resource<constant>{%[[SIZE]]} => !stream.timepoint
  %result, %result_ready = stream.async.parameter.load on(#hal.device.affinity<@device>) %scope::%key[%offset] : !stream.resource<constant>{%size} => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<constant>{%[[SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<constant>{%size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @asyncParameterLoadWithAwait
// CHECK-SAME: (%[[AWAIT:.+]]: !stream.timepoint, %[[SIZE:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func private @asyncParameterLoadWithAwait(%await: !stream.timepoint, %size: index, %scope: !util.buffer, %key: !util.buffer) -> !stream.resource<constant> {
  // CHECK: %[[OFFSET:.+]] = arith.constant 200000000
  %offset = arith.constant 200000000 : i64
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.load await(%[[AWAIT]]) %[[SCOPE]]::%[[KEY]][%[[OFFSET]]] : !stream.resource<constant>{%[[SIZE]]} => !stream.timepoint
  %result, %result_ready = stream.async.parameter.load await(%await) %scope::%key[%offset] : !stream.resource<constant>{%size} => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<constant>{%[[SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<constant>{%size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @asyncParameterRead
// CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func private @asyncParameterRead(%target: !stream.resource<transient>, %size: index, %scope: !util.buffer, %key: !util.buffer) -> !stream.resource<transient> {
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 256
  %param_offset = arith.constant 256 : i64
  // CHECK-DAG: %[[TARGET_OFFSET:.+]] = arith.constant 64
  %target_offset = arith.constant 64 : index
  // CHECK-DAG: %[[TARGET_END:.+]] = arith.constant 576
  %target_end = arith.constant 576 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 512
  %length = arith.constant 512 : index
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.read %[[SCOPE]]::%[[KEY]][%[[PARAM_OFFSET]]] -> %[[TARGET]][%[[TARGET_OFFSET]] to %[[TARGET_END]] for %[[LENGTH]]] : !stream.resource<transient>{%[[SIZE]]} => !stream.timepoint
  %result, %result_ready = stream.async.parameter.read %scope::%key[%param_offset] -> %target[%target_offset to %target_end for %length] : !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<transient>{%[[SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @asyncParameterReadNoScope
// CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[KEY:.+]]: !util.buffer)
util.func private @asyncParameterReadNoScope(%target: !stream.resource<transient>, %size: index, %key: !util.buffer) -> !stream.resource<transient> {
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 1024
  %param_offset = arith.constant 1024 : i64
  // CHECK-DAG: %[[TARGET_OFFSET:.+]] = arith.constant 128
  %target_offset = arith.constant 128 : index
  // CHECK-DAG: %[[TARGET_END:.+]] = arith.constant 384
  %target_end = arith.constant 384 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 256
  %length = arith.constant 256 : index
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.read %[[KEY]][%[[PARAM_OFFSET]]] -> %[[TARGET]][%[[TARGET_OFFSET]] to %[[TARGET_END]] for %[[LENGTH]]] : !stream.resource<transient>{%[[SIZE]]} => !stream.timepoint
  %result, %result_ready = stream.async.parameter.read %key[%param_offset] -> %target[%target_offset to %target_end for %length] : !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<transient>{%[[SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @asyncParameterReadWithAffinity
// CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func private @asyncParameterReadWithAffinity(%target: !stream.resource<transient>, %size: index, %scope: !util.buffer, %key: !util.buffer) -> !stream.resource<transient> {
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 2048
  %param_offset = arith.constant 2048 : i64
  // CHECK-DAG: %[[TARGET_OFFSET:.+]] = arith.constant 0
  %target_offset = arith.constant 0 : index
  // CHECK-DAG: %[[TARGET_END:.+]] = arith.constant 1024
  %target_end = arith.constant 1024 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 1024
  %length = arith.constant 1024 : index
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.read on(#hal.device.affinity<@device>) %[[SCOPE]]::%[[KEY]][%[[PARAM_OFFSET]]] -> %[[TARGET]][%[[TARGET_OFFSET]] to %[[TARGET_END]] for %[[LENGTH]]] : !stream.resource<transient>{%[[SIZE]]} => !stream.timepoint
  %result, %result_ready = stream.async.parameter.read on(#hal.device.affinity<@device>) %scope::%key[%param_offset] -> %target[%target_offset to %target_end for %length] : !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<transient>{%[[SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @asyncParameterWrite
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func private @asyncParameterWrite(%source: !stream.resource<transient>, %size: index, %scope: !util.buffer, %key: !util.buffer) -> !stream.resource<transient> {
  // CHECK-DAG: %[[SOURCE_OFFSET:.+]] = arith.constant 192
  %source_offset = arith.constant 192 : index
  // CHECK-DAG: %[[SOURCE_END:.+]] = arith.constant 960
  %source_end = arith.constant 960 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 768
  %length = arith.constant 768 : index
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 4096
  %param_offset = arith.constant 4096 : i64
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.write %[[SOURCE]][%[[SOURCE_OFFSET]] to %[[SOURCE_END]] for %[[LENGTH]]] -> %[[SCOPE]]::%[[KEY]][%[[PARAM_OFFSET]]] : !stream.resource<transient>{%[[SIZE]]} => !stream.timepoint
  %result, %result_ready = stream.async.parameter.write %source[%source_offset to %source_end for %length] -> %scope::%key[%param_offset] : !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<transient>{%[[SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @asyncParameterWriteNoScope
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[KEY:.+]]: !util.buffer)
util.func private @asyncParameterWriteNoScope(%source: !stream.resource<transient>, %size: index, %key: !util.buffer) -> !stream.resource<transient> {
  // CHECK-DAG: %[[SOURCE_OFFSET:.+]] = arith.constant 320
  %source_offset = arith.constant 320 : index
  // CHECK-DAG: %[[SOURCE_END:.+]] = arith.constant 704
  %source_end = arith.constant 704 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 384
  %length = arith.constant 384 : index
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 8192
  %param_offset = arith.constant 8192 : i64
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.write %[[SOURCE]][%[[SOURCE_OFFSET]] to %[[SOURCE_END]] for %[[LENGTH]]] -> %[[KEY]][%[[PARAM_OFFSET]]] : !stream.resource<transient>{%[[SIZE]]} => !stream.timepoint
  %result, %result_ready = stream.async.parameter.write %source[%source_offset to %source_end for %length] -> %key[%param_offset] : !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<transient>{%[[SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @asyncParameterWriteWithAffinity
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func private @asyncParameterWriteWithAffinity(%source: !stream.resource<transient>, %size: index, %scope: !util.buffer, %key: !util.buffer) -> !stream.resource<transient> {
  // CHECK-DAG: %[[SOURCE_OFFSET:.+]] = arith.constant 448
  %source_offset = arith.constant 448 : index
  // CHECK-DAG: %[[SOURCE_END:.+]] = arith.constant 1344
  %source_end = arith.constant 1344 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 896
  %length = arith.constant 896 : index
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 16384
  %param_offset = arith.constant 16384 : i64
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.write on(#hal.device.affinity<@device>) %[[SOURCE]][%[[SOURCE_OFFSET]] to %[[SOURCE_END]] for %[[LENGTH]]] -> %[[SCOPE]]::%[[KEY]][%[[PARAM_OFFSET]]] : !stream.resource<transient>{%[[SIZE]]} => !stream.timepoint
  %result, %result_ready = stream.async.parameter.write on(#hal.device.affinity<@device>) %source[%source_offset to %source_end for %length] -> %scope::%key[%param_offset] : !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<transient>{%[[SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @asyncParameterGather
// CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer, %[[KEY2:.+]]: !util.buffer)
util.func private @asyncParameterGather(%target: !stream.resource<transient>, %size: index, %scope: !util.buffer, %key0: !util.buffer, %key1: !util.buffer, %key2: !util.buffer) -> !stream.resource<transient> {
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 512
  %param_offset0 = arith.constant 512 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 1536
  %param_offset1 = arith.constant 1536 : i64
  // CHECK-DAG: %[[PARAM_OFFSET2:.+]] = arith.constant 3072
  %param_offset2 = arith.constant 3072 : i64
  // CHECK-DAG: %[[TARGET_OFFSET0:.+]] = arith.constant 0
  %target_offset0 = arith.constant 0 : index
  // CHECK-DAG: %[[TARGET_OFFSET1:.+]] = arith.constant 256
  %target_offset1 = arith.constant 256 : index
  // CHECK-DAG: %[[TARGET_OFFSET2:.+]] = arith.constant 512
  %target_offset2 = arith.constant 512 : index
  // CHECK-DAG: %[[TARGET_END0:.+]] = arith.constant 128
  %target_end0 = arith.constant 128 : index
  // CHECK-DAG: %[[TARGET_END1:.+]] = arith.constant 512
  %target_end1 = arith.constant 512 : index
  // CHECK-DAG: %[[TARGET_END2:.+]] = arith.constant 704
  %target_end2 = arith.constant 704 : index
  // CHECK-DAG: %[[LENGTH0:.+]] = arith.constant 128
  %length0 = arith.constant 128 : index
  // CHECK-DAG: %[[LENGTH1:.+]] = arith.constant 256
  %length1 = arith.constant 256 : index
  // CHECK-DAG: %[[LENGTH2:.+]] = arith.constant 192
  %length2 = arith.constant 192 : index
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.gather {
  // CHECK-NEXT: %[[SCOPE]]::%[[KEY0]][%[[PARAM_OFFSET0]]] -> %[[TARGET]][%[[TARGET_OFFSET0]] to %[[TARGET_END0]] for %[[LENGTH0]]] : !stream.resource<transient>{%[[SIZE]]},
  // CHECK-NEXT: %[[SCOPE]]::%[[KEY1]][%[[PARAM_OFFSET1]]] -> %[[TARGET]][%[[TARGET_OFFSET1]] to %[[TARGET_END1]] for %[[LENGTH1]]] : !stream.resource<transient>{%[[SIZE]]},
  // CHECK-NEXT: %[[SCOPE]]::%[[KEY2]][%[[PARAM_OFFSET2]]] -> %[[TARGET]][%[[TARGET_OFFSET2]] to %[[TARGET_END2]] for %[[LENGTH2]]] : !stream.resource<transient>{%[[SIZE]]}
  // CHECK-NEXT: } : !stream.resource<transient> => !stream.timepoint
  %result, %result_ready = stream.async.parameter.gather {
    %scope::%key0[%param_offset0] -> %target[%target_offset0 to %target_end0 for %length0] : !stream.resource<transient>{%size},
    %scope::%key1[%param_offset1] -> %target[%target_offset1 to %target_end1 for %length1] : !stream.resource<transient>{%size},
    %scope::%key2[%param_offset2] -> %target[%target_offset2 to %target_end2 for %length2] : !stream.resource<transient>{%size}
  } : !stream.resource<transient> => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<transient>{%[[SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @asyncParameterGatherNoScope
// CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer)
util.func private @asyncParameterGatherNoScope(%target: !stream.resource<transient>, %size: index, %key0: !util.buffer, %key1: !util.buffer) -> !stream.resource<transient> {
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 6144
  %param_offset0 = arith.constant 6144 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 7168
  %param_offset1 = arith.constant 7168 : i64
  // CHECK-DAG: %[[TARGET_OFFSET0:.+]] = arith.constant 100
  %target_offset0 = arith.constant 100 : index
  // CHECK-DAG: %[[TARGET_OFFSET1:.+]] = arith.constant 400
  %target_offset1 = arith.constant 400 : index
  // CHECK-DAG: %[[TARGET_END0:.+]] = arith.constant 300
  %target_end0 = arith.constant 300 : index
  // CHECK-DAG: %[[TARGET_END1:.+]] = arith.constant 700
  %target_end1 = arith.constant 700 : index
  // CHECK-DAG: %[[LENGTH0:.+]] = arith.constant 200
  %length0 = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH1:.+]] = arith.constant 300
  %length1 = arith.constant 300 : index
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.gather {
  // CHECK-NEXT: %[[KEY0]][%[[PARAM_OFFSET0]]] -> %[[TARGET]][%[[TARGET_OFFSET0]] to %[[TARGET_END0]] for %[[LENGTH0]]] : !stream.resource<transient>{%[[SIZE]]},
  // CHECK-NEXT: %[[KEY1]][%[[PARAM_OFFSET1]]] -> %[[TARGET]][%[[TARGET_OFFSET1]] to %[[TARGET_END1]] for %[[LENGTH1]]] : !stream.resource<transient>{%[[SIZE]]}
  // CHECK-NEXT: } : !stream.resource<transient> => !stream.timepoint
  %result, %result_ready = stream.async.parameter.gather {
    %key0[%param_offset0] -> %target[%target_offset0 to %target_end0 for %length0] : !stream.resource<transient>{%size},
    %key1[%param_offset1] -> %target[%target_offset1 to %target_end1 for %length1] : !stream.resource<transient>{%size}
  } : !stream.resource<transient> => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<transient>{%[[SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @asyncParameterGatherWithAffinity
// CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer)
util.func private @asyncParameterGatherWithAffinity(%target: !stream.resource<transient>, %size: index, %scope: !util.buffer, %key0: !util.buffer, %key1: !util.buffer) -> !stream.resource<transient> {
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 12288
  %param_offset0 = arith.constant 12288 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 13312
  %param_offset1 = arith.constant 13312 : i64
  // CHECK-DAG: %[[TARGET_OFFSET0:.+]] = arith.constant 0
  %target_offset0 = arith.constant 0 : index
  // CHECK-DAG: %[[TARGET_OFFSET1:.+]] = arith.constant 512
  %target_offset1 = arith.constant 512 : index
  // CHECK-DAG: %[[TARGET_END0:.+]] = arith.constant 512
  %target_end0 = arith.constant 512 : index
  // CHECK-DAG: %[[TARGET_END1:.+]] = arith.constant 1024
  %target_end1 = arith.constant 1024 : index
  // CHECK-DAG: %[[LENGTH0:.+]] = arith.constant 512
  %length0 = arith.constant 512 : index
  // CHECK-DAG: %[[LENGTH1:.+]] = arith.constant 512
  %length1 = arith.constant 512 : index
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.gather on(#hal.device.affinity<@device>) {
  // CHECK-NEXT: %[[SCOPE]]::%[[KEY0]][%[[PARAM_OFFSET0]]] -> %[[TARGET]][%[[TARGET_OFFSET0]] to %[[TARGET_END0]] for %[[LENGTH0]]] : !stream.resource<transient>{%[[SIZE]]},
  // CHECK-NEXT: %[[SCOPE]]::%[[KEY1]][%[[PARAM_OFFSET1]]] -> %[[TARGET]][%[[TARGET_OFFSET1]] to %[[TARGET_END1]] for %[[LENGTH1]]] : !stream.resource<transient>{%[[SIZE]]}
  // CHECK-NEXT: } : !stream.resource<transient> => !stream.timepoint
  %result, %result_ready = stream.async.parameter.gather on(#hal.device.affinity<@device>) {
    %scope::%key0[%param_offset0] -> %target[%target_offset0 to %target_end0 for %length0] : !stream.resource<transient>{%size},
    %scope::%key1[%param_offset1] -> %target[%target_offset1 to %target_end1 for %length1] : !stream.resource<transient>{%size}
  } : !stream.resource<transient> => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<transient>{%[[SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @asyncParameterScatter
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer, %[[KEY2:.+]]: !util.buffer)
util.func private @asyncParameterScatter(%source: !stream.resource<transient>, %size: index, %scope: !util.buffer, %key0: !util.buffer, %key1: !util.buffer, %key2: !util.buffer) -> !stream.resource<transient> {
  // CHECK-DAG: %[[SOURCE_OFFSET0:.+]] = arith.constant 0
  %source_offset0 = arith.constant 0 : index
  // CHECK-DAG: %[[SOURCE_OFFSET1:.+]] = arith.constant 384
  %source_offset1 = arith.constant 384 : index
  // CHECK-DAG: %[[SOURCE_OFFSET2:.+]] = arith.constant 768
  %source_offset2 = arith.constant 768 : index
  // CHECK-DAG: %[[SOURCE_END0:.+]] = arith.constant 384
  %source_end0 = arith.constant 384 : index
  // CHECK-DAG: %[[SOURCE_END1:.+]] = arith.constant 768
  %source_end1 = arith.constant 768 : index
  // CHECK-DAG: %[[SOURCE_END2:.+]] = arith.constant 1152
  %source_end2 = arith.constant 1152 : index
  // CHECK-DAG: %[[LENGTH0:.+]] = arith.constant 384
  %length0 = arith.constant 384 : index
  // CHECK-DAG: %[[LENGTH1:.+]] = arith.constant 384
  %length1 = arith.constant 384 : index
  // CHECK-DAG: %[[LENGTH2:.+]] = arith.constant 384
  %length2 = arith.constant 384 : index
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 20480
  %param_offset0 = arith.constant 20480 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 21504
  %param_offset1 = arith.constant 21504 : i64
  // CHECK-DAG: %[[PARAM_OFFSET2:.+]] = arith.constant 22528
  %param_offset2 = arith.constant 22528 : i64
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.scatter {
  // CHECK-NEXT: %[[SOURCE]][%[[SOURCE_OFFSET0]] to %[[SOURCE_END0]] for %[[LENGTH0]]] : !stream.resource<transient>{%[[SIZE]]} -> %[[SCOPE]]::%[[KEY0]][%[[PARAM_OFFSET0]]],
  // CHECK-NEXT: %[[SOURCE]][%[[SOURCE_OFFSET1]] to %[[SOURCE_END1]] for %[[LENGTH1]]] : !stream.resource<transient>{%[[SIZE]]} -> %[[SCOPE]]::%[[KEY1]][%[[PARAM_OFFSET1]]],
  // CHECK-NEXT: %[[SOURCE]][%[[SOURCE_OFFSET2]] to %[[SOURCE_END2]] for %[[LENGTH2]]] : !stream.resource<transient>{%[[SIZE]]} -> %[[SCOPE]]::%[[KEY2]][%[[PARAM_OFFSET2]]]
  // CHECK-NEXT: } : !stream.resource<transient> => !stream.timepoint
  %result, %result_ready = stream.async.parameter.scatter {
    %source[%source_offset0 to %source_end0 for %length0] : !stream.resource<transient>{%size} -> %scope::%key0[%param_offset0],
    %source[%source_offset1 to %source_end1 for %length1] : !stream.resource<transient>{%size} -> %scope::%key1[%param_offset1],
    %source[%source_offset2 to %source_end2 for %length2] : !stream.resource<transient>{%size} -> %scope::%key2[%param_offset2]
  } : !stream.resource<transient> => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<transient>{%[[SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @asyncParameterScatterNoScope
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer)
util.func private @asyncParameterScatterNoScope(%source: !stream.resource<transient>, %size: index, %key0: !util.buffer, %key1: !util.buffer) -> !stream.resource<transient> {
  // CHECK-DAG: %[[SOURCE_OFFSET0:.+]] = arith.constant 128
  %source_offset0 = arith.constant 128 : index
  // CHECK-DAG: %[[SOURCE_OFFSET1:.+]] = arith.constant 640
  %source_offset1 = arith.constant 640 : index
  // CHECK-DAG: %[[SOURCE_END0:.+]] = arith.constant 640
  %source_end0 = arith.constant 640 : index
  // CHECK-DAG: %[[SOURCE_END1:.+]] = arith.constant 1152
  %source_end1 = arith.constant 1152 : index
  // CHECK-DAG: %[[LENGTH0:.+]] = arith.constant 512
  %length0 = arith.constant 512 : index
  // CHECK-DAG: %[[LENGTH1:.+]] = arith.constant 512
  %length1 = arith.constant 512 : index
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 24576
  %param_offset0 = arith.constant 24576 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 25600
  %param_offset1 = arith.constant 25600 : i64
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.scatter {
  // CHECK-NEXT: %[[SOURCE]][%[[SOURCE_OFFSET0]] to %[[SOURCE_END0]] for %[[LENGTH0]]] : !stream.resource<transient>{%[[SIZE]]} -> %[[KEY0]][%[[PARAM_OFFSET0]]],
  // CHECK-NEXT: %[[SOURCE]][%[[SOURCE_OFFSET1]] to %[[SOURCE_END1]] for %[[LENGTH1]]] : !stream.resource<transient>{%[[SIZE]]} -> %[[KEY1]][%[[PARAM_OFFSET1]]]
  // CHECK-NEXT: } : !stream.resource<transient> => !stream.timepoint
  %result, %result_ready = stream.async.parameter.scatter {
    %source[%source_offset0 to %source_end0 for %length0] : !stream.resource<transient>{%size} -> %key0[%param_offset0],
    %source[%source_offset1 to %source_end1 for %length1] : !stream.resource<transient>{%size} -> %key1[%param_offset1]
  } : !stream.resource<transient> => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<transient>{%[[SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @asyncParameterScatterWithAffinity
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer)
util.func private @asyncParameterScatterWithAffinity(%source: !stream.resource<transient>, %size: index, %scope: !util.buffer, %key0: !util.buffer, %key1: !util.buffer) -> !stream.resource<transient> {
  // CHECK-DAG: %[[SOURCE_OFFSET0:.+]] = arith.constant 0
  %source_offset0 = arith.constant 0 : index
  // CHECK-DAG: %[[SOURCE_OFFSET1:.+]] = arith.constant 1024
  %source_offset1 = arith.constant 1024 : index
  // CHECK-DAG: %[[SOURCE_END0:.+]] = arith.constant 1024
  %source_end0 = arith.constant 1024 : index
  // CHECK-DAG: %[[SOURCE_END1:.+]] = arith.constant 2048
  %source_end1 = arith.constant 2048 : index
  // CHECK-DAG: %[[LENGTH0:.+]] = arith.constant 1024
  %length0 = arith.constant 1024 : index
  // CHECK-DAG: %[[LENGTH1:.+]] = arith.constant 1024
  %length1 = arith.constant 1024 : index
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 32768
  %param_offset0 = arith.constant 32768 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 34816
  %param_offset1 = arith.constant 34816 : i64
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.scatter on(#hal.device.affinity<@device>) {
  // CHECK-NEXT: %[[SOURCE]][%[[SOURCE_OFFSET0]] to %[[SOURCE_END0]] for %[[LENGTH0]]] : !stream.resource<transient>{%[[SIZE]]} -> %[[SCOPE]]::%[[KEY0]][%[[PARAM_OFFSET0]]],
  // CHECK-NEXT: %[[SOURCE]][%[[SOURCE_OFFSET1]] to %[[SOURCE_END1]] for %[[LENGTH1]]] : !stream.resource<transient>{%[[SIZE]]} -> %[[SCOPE]]::%[[KEY1]][%[[PARAM_OFFSET1]]]
  // CHECK-NEXT: } : !stream.resource<transient> => !stream.timepoint
  %result, %result_ready = stream.async.parameter.scatter on(#hal.device.affinity<@device>) {
    %source[%source_offset0 to %source_end0 for %length0] : !stream.resource<transient>{%size} -> %scope::%key0[%param_offset0],
    %source[%source_offset1 to %source_end1 for %length1] : !stream.resource<transient>{%size} -> %scope::%key1[%param_offset1]
  } : !stream.resource<transient> => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<transient>{%[[SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// Metadata operations are allowed on resources from timeline ops.
// CHECK-LABEL: @asyncParameterLoadLegalSubviewMetadata
// CHECK-SAME: (%[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func private @asyncParameterLoadLegalSubviewMetadata(%scope: !util.buffer, %key: !util.buffer) {
  %c0_i64 = arith.constant 0 : i64
  %c100 = arith.constant 100 : index
  %c20 = arith.constant 20 : index
  %c50 = arith.constant 50 : index
  %loaded, %loaded_ready = stream.async.parameter.load %scope::%key[%c0_i64] : !stream.resource<constant>{%c100} => !stream.timepoint
  // CHECK: stream.resource.subview
  %subview = stream.resource.subview %loaded[%c20] : !stream.resource<constant>{%c100} -> !stream.resource<constant>{%c50}
  util.return
}

// -----

// Proper await synchronization makes resource timeline-safe.
// CHECK-LABEL: @asyncParameterLoadLegalAwaitThenSlice
// CHECK-SAME: (%[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func private @asyncParameterLoadLegalAwaitThenSlice(%scope: !util.buffer, %key: !util.buffer) -> !stream.resource<constant> {
  %c0_i64 = arith.constant 0 : i64
  %c100 = arith.constant 100 : index
  %c10 = arith.constant 10 : index
  %c60 = arith.constant 60 : index
  %c50 = arith.constant 50 : index
  %loaded, %loaded_ready = stream.async.parameter.load %scope::%key[%c0_i64] : !stream.resource<constant>{%c100} => !stream.timepoint
  %awaited = stream.timepoint.await %loaded_ready => %loaded : !stream.resource<constant>{%c100}
  // NO ERROR: awaited resource is timeline-safe.
  // CHECK: stream.async.slice
  %sliced = stream.async.slice %awaited[%c10 to %c60] : !stream.resource<constant>{%c100} -> !stream.resource<constant>{%c50}
  util.return %sliced : !stream.resource<constant>
}

// -----

// Explicit await_timepoint synchronization makes operation legal.
// CHECK-LABEL: @asyncParameterLoadLegalExplicitAwait
// CHECK-SAME: (%[[SCOPE:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer, %[[KEY2:.+]]: !util.buffer)
util.func private @asyncParameterLoadLegalExplicitAwait(%scope: !util.buffer, %key1: !util.buffer, %key2: !util.buffer) -> !stream.resource<constant> {
  %c0_i64 = arith.constant 0 : i64
  %c100_i64 = arith.constant 100 : i64
  %c100 = arith.constant 100 : index
  %c0 = arith.constant 0 : index
  %c50 = arith.constant 50 : index
  %loaded, %loaded_ready = stream.async.parameter.load %scope::%key1[%c0_i64] : !stream.resource<constant>{%c100} => !stream.timepoint
  // NO ERROR: explicit await_timepoint synchronization via await(%loaded_ready).
  // CHECK: stream.async.parameter.read
  %result, %result_ready = stream.async.parameter.read await(%loaded_ready) %scope::%key2[%c100_i64] -> %loaded[%c0 to %c50 for %c50] : !stream.resource<constant>{%c100} => !stream.timepoint
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<constant>{%c100}
  util.return %result_sync : !stream.resource<constant>
}

// -----

// Legal chain: load -> await -> subview (metadata) -> slice (streamable).
// CHECK-LABEL: @asyncParameterLoadLegalAwaitSubviewSlice
// CHECK-SAME: (%[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func private @asyncParameterLoadLegalAwaitSubviewSlice(%scope: !util.buffer, %key: !util.buffer) -> !stream.resource<constant> {
  %c0_i64 = arith.constant 0 : i64
  %c100 = arith.constant 100 : index
  %c20 = arith.constant 20 : index
  %c70 = arith.constant 70 : index
  %c50 = arith.constant 50 : index
  %c10 = arith.constant 10 : index
  %c40 = arith.constant 40 : index
  %c30 = arith.constant 30 : index
  %loaded, %loaded_ready = stream.async.parameter.load %scope::%key[%c0_i64] : !stream.resource<constant>{%c100} => !stream.timepoint
  %awaited = stream.timepoint.await %loaded_ready => %loaded : !stream.resource<constant>{%c100}
  // Subview of awaited resource is legal (metadata operation).
  %subview = stream.resource.subview %awaited[%c20] : !stream.resource<constant>{%c100} -> !stream.resource<constant>{%c50}
  // NO ERROR: subview is of awaited (timeline-safe) resource.
  // CHECK: stream.async.slice
  %sliced = stream.async.slice %subview[%c10 to %c40] : !stream.resource<constant>{%c50} -> !stream.resource<constant>{%c30}
  util.return %sliced : !stream.resource<constant>
}
