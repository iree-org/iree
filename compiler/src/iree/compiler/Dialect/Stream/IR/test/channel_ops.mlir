// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @channel_create
//  CHECK-SAME: (%[[RANK:.+]]: index, %[[COUNT:.+]]: index)
util.func private @channel_create(%rank: index, %count: index) {
  // CHECK: %channel = stream.channel.create on(#hal.affinity.queue<[0, 1]>) rank(%[[RANK]]) count(%[[COUNT]]) : !stream.channel
  %channel = stream.channel.create on(#hal.affinity.queue<[0, 1]>) rank(%rank) count(%count) : !stream.channel
  util.return
}

// -----

// CHECK-LABEL: @channel_split
//  CHECK-SAME: (%[[BASE_CHANNEL:.+]]: !stream.channel)
util.func private @channel_split(%base_channel: !stream.channel) {
  // CHECK-DAG: %[[COLOR:.+]] = arith.constant 100 : index
  %color = arith.constant 100 : index
  // CHECK-DAG: %[[KEY:.+]] = arith.constant 101 : index
  %key = arith.constant 101 : index
  // CHECK: %channel = stream.channel.split %[[BASE_CHANNEL]], %[[COLOR]], %[[KEY]] : !stream.channel -> !stream.channel
  %split_channel = stream.channel.split %base_channel, %color, %key : !stream.channel -> !stream.channel
  util.return
}

// -----

// CHECK-LABEL: @channel_rank
//  CHECK-SAME: (%[[CHANNEL:.+]]: !stream.channel)
util.func private @channel_rank(%channel: !stream.channel) -> index {
  // CHECK: = stream.channel.rank %[[CHANNEL]] : index
  %rank = stream.channel.rank %channel : index
  util.return %rank : index
}

// -----

// CHECK-LABEL: @channel_count
//  CHECK-SAME: (%[[CHANNEL:.+]]: !stream.channel)
util.func private @channel_count(%channel: !stream.channel) -> index {
  // CHECK: = stream.channel.count %[[CHANNEL]] : index
  %count = stream.channel.count %channel : index
  util.return %count : index
}
