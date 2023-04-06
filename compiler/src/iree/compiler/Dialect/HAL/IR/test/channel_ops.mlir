// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @channel_default
//  CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64, %[[ID:.+]]: !util.buffer, %[[GROUP:.+]]: !util.buffer)
func.func @channel_default(%device: !hal.device, %affinity: i64, %id: !util.buffer, %group: !util.buffer) {
  //      CHECK: %channel = hal.channel.default
  // CHECK-SAME:   device(%[[DEVICE]] : !hal.device)
  // CHECK-SAME:   affinity(%[[AFFINITY]])
  // CHECK-SAME:   flags(0)
  // CHECK-SAME:   id(%[[ID]])
  // CHECK-SAME:   group(%[[GROUP]]) : !hal.channel
  %channel = hal.channel.default device(%device : !hal.device)
                               affinity(%affinity)
                                  flags(0)
                                     id(%id)
                                  group(%group) : !hal.channel
  return
}

// -----

// CHECK-LABEL: @channel_rank_and_count
// CHECK-SAME: (%[[CHANNEL:.+]]: !hal.channel)
func.func @channel_rank_and_count(%channel: !hal.channel) -> (i32, i32) {
  // CHECK: = hal.channel.rank_and_count<%[[CHANNEL]] : !hal.channel> : i32, i32
  %rank, %count = hal.channel.rank_and_count<%channel : !hal.channel> : i32, i32
  return %rank, %count : i32, i32
}
