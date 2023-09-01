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
  %channel = stream.channel.create on(#hal.affinity.queue<[0, 1]>) group("group") : !stream.channel
  // CHECK: return %[[CHANNEL]]
  return %channel : !stream.channel
}

// -----

// CHECK-LABEL: @channel_split
//  CHECK-SAME: (%[[BASE_CHANNEL:.+]]: !hal.channel)
func.func @channel_split(%base_channel: !stream.channel) {
  // CHECK-DAG: %[[COLOR_INDEX:.+]] = arith.constant 100
  %color = arith.constant 100 : index
  // CHECK-DAG: %[[KEY_INDEX:.+]] = arith.constant 101
  %key = arith.constant 101 : index
  // CHECK-DAG: %[[COLOR_I32:.+]] = arith.index_cast %[[COLOR_INDEX]] : index to i32
  // CHECK-DAG: %[[KEY_I32:.+]] = arith.index_cast %[[KEY_INDEX]] : index to i32
  // CHECK: %channel = hal.channel.split<%[[BASE_CHANNEL]] : !hal.channel> color(%[[COLOR_I32]]) key(%[[KEY_I32]]) flags(0) : !hal.channel
  %split_channel = stream.channel.split %base_channel, %color, %key : !stream.channel -> !stream.channel
  return
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
