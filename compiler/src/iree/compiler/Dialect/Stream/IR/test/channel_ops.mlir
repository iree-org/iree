// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @channel_default
func.func @channel_default() {
  // CHECK: %channel_default = stream.channel.default on(#hal.affinity.queue<[0, 1]>) : !stream.channel
  %channel = stream.channel.default on(#hal.affinity.queue<[0, 1]>) : !stream.channel
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
