// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @channel_default
func.func @channel_default() {
  // CHECK: %channel = stream.channel.default on(#hal.affinity.queue<[0, 1]>) : !stream.channel
  %channel = stream.channel.default on(#hal.affinity.queue<[0, 1]>) : !stream.channel
  return
}

// -----

// CHECK-LABEL: @channel_create
//  CHECK-SAME: (%[[RANK:.+]]: index, %[[COUNT:.+]]: index)
func.func @channel_create(%rank: index, %count: index) {
  // CHECK: %channel = stream.channel.create on(#hal.affinity.queue<[0, 1]>) rank(%[[RANK]]) count(%[[COUNT]]) : !stream.channel
  %channel = stream.channel.create on(#hal.affinity.queue<[0, 1]>) rank(%rank) count(%count) : !stream.channel
  return
}

// -----

// CHECK-LABEL: @channel_rank
//  CHECK-SAME: (%[[CHANNEL:.+]]: !stream.channel)
func.func @channel_rank(%channel: !stream.channel) -> index {
  // CHECK: = stream.channel.rank %[[CHANNEL]] : index
  %rank = stream.channel.rank %channel : index
  return %rank : index
}

// -----

// CHECK-LABEL: @channel_count
//  CHECK-SAME: (%[[CHANNEL:.+]]: !stream.channel)
func.func @channel_count(%channel: !stream.channel) -> index {
  // CHECK: = stream.channel.count %[[CHANNEL]] : index
  %count = stream.channel.count %channel : index
  return %count : index
}
