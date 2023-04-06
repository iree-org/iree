// RUN: iree-opt --split-input-file --iree-hal-conversion %s | FileCheck %s

// CHECK-LABEL: @channel_create
//  CHECK-SAME: () -> !hal.channel
func.func @channel_create() -> !stream.channel {
  // CHECK-DAG: %[[DEVICE:.+]] = hal.ex.shared_device : !hal.device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant 3
  // CHECK-DAG: %[[ID:.+]] = util.null : !util.buffer
  // CHECK-DAG: %[[GROUP:.+]] = util.buffer.constant : !util.buffer = "group"
  // CHECK-DAG: %[[DEFAULT:.+]] = arith.constant -1
  // CHECK: %[[CHANNEL:.+]] = hal.channel.create device(%[[DEVICE]] : !hal.device) affinity(%[[AFFINITY]]) flags(0) id(%[[ID]]) group(%[[GROUP]]) rank(%[[DEFAULT]]) count(%[[DEFAULT]]) : !hal.channel
  %channel = stream.channel.default on(#hal.affinity.queue<[0, 1]>) group("group") : !stream.channel
  // CHECK: return %[[CHANNEL]]
  return %channel : !stream.channel
}

// -----

// CHECK-LABEL: @channel_default
//  CHECK-SAME: () -> !hal.channel
func.func @channel_default() -> !stream.channel {
  // CHECK-DAG: %[[DEVICE:.+]] = hal.ex.shared_device : !hal.device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant 3
  // CHECK-DAG: %[[ID:.+]] = util.null : !util.buffer
  // CHECK-DAG: %[[GROUP:.+]] = util.buffer.constant : !util.buffer = "group"
  // CHECK-DAG: %[[DEFAULT:.+]] = arith.constant -1
  // CHECK: %[[CHANNEL:.+]] = hal.channel.create device(%[[DEVICE]] : !hal.device) affinity(%[[AFFINITY]]) flags(0) id(%[[ID]]) group(%[[GROUP]]) rank(%[[DEFAULT]]) count(%[[DEFAULT]]) : !hal.channel
  %channel = stream.channel.default on(#hal.affinity.queue<[0, 1]>) group("group") : !stream.channel
  // CHECK: return %[[CHANNEL]]
  return %channel : !stream.channel
}

// -----

// CHECK-LABEL: @channel_rank
//  CHECK-SAME: (%[[CHANNEL:.+]]: !hal.channel)
func.func @channel_rank(%channel: !stream.channel) -> index {
  // CHECK: %[[RANK_I32:.+]], %[[COUNT_I32:.+]] = hal.channel.rank_and_count<%[[CHANNEL]] : !hal.channel> : i32, i32
  // CHECK: %[[RANK:.+]] = arith.index_cast %[[RANK_I32]] : i32 to index
  %rank = stream.channel.rank %channel : index
  // CHECK: return %[[RANK]]
  return %rank : index
}

// -----

// CHECK-LABEL: @channel_count
//  CHECK-SAME: (%[[CHANNEL:.+]]: !hal.channel) -> index
func.func @channel_count(%channel: !stream.channel) -> index {
  // CHECK: %[[RANK_I32:.+]], %[[COUNT_I32:.+]] = hal.channel.rank_and_count<%[[CHANNEL]] : !hal.channel> : i32, i32
  // CHECK: %[[COUNT:.+]] = arith.index_cast %[[COUNT_I32]] : i32 to index
  %count = stream.channel.count %channel : index
  // CHECK: return %[[COUNT]]
  return %count : index
}

// -----

// CHECK-LABEL: @channel_create_split
//  CHECK-SAME: () -> !hal.channel
func.func @channel_create_split() -> !stream.channel {
  // CHECK-DAG: %[[DEVICE:.+]] = hal.ex.shared_device : !hal.device
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant 3
  // CHECK-DAG: %[[ID:.+]] = util.null : !util.buffer
  // CHECK-DAG: %[[GROUP:.+]] = util.null : !util.buffer
  // CHECK-DAG: %[[DEFAULT:.+]] = arith.constant -1
  // CHECK: %[[CHANNEL:.+]] = hal.channel.create device(%[[DEVICE]] : !hal.device) affinity(%[[AFFINITY]]) flags(0) id(%[[ID]]) group(%[[GROUP]]) rank(%[[DEFAULT]]) count(%[[DEFAULT]]) : !hal.channel
  // CHECK-DAG: %[[DEVICE_0:.+]] = hal.ex.shared_device : !hal.device
  // CHECK-DAG: %[[DEFAULT_0:.+]] = arith.constant -1
  // CHECK-DAG: %[[GROUPS:.+]] = util.buffer.constant : !util.buffer = "(0),(1)"
  // CHECK: %[[CHANNEL_SPLIT:.+]] = hal.channel.split device(%[[DEVICE_0]] : !hal.device) affinity(%[[DEFAULT_0]]) groups(%[[GROUPS]]) %[[CHANNEL]] : !hal.channel
  %channel_default = stream.channel.default on(#hal.affinity.queue<[0, 1]>) : !stream.channel
  %channel_split = stream.channel.split groups("(0),(1)") %channel_default : !stream.channel

  // CHECK: return %[[CHANNEL_SPLIT]]
  return %channel_split : !stream.channel
}
