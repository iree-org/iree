// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @channel_create
//  CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64, %[[ID:.+]]: !util.buffer, %[[GROUP:.+]]: !util.buffer, %[[RANK:.+]]: i32, %[[COUNT:.+]]: i32)
func.func @channel_create(%device: !hal.device, %affinity: i64, %id: !util.buffer, %group: !util.buffer, %rank: i32, %count: i32) {
  //      CHECK: %channel = hal.channel.create
  // CHECK-SAME:   device(%[[DEVICE]] : !hal.device)
  // CHECK-SAME:   affinity(%[[AFFINITY]])
  // CHECK-SAME:   flags(0)
  // CHECK-SAME:   id(%[[ID]])
  // CHECK-SAME:   group(%[[GROUP]])
  // CHECK-SAME:   rank(%[[RANK]])
  // CHECK-SAME:   count(%[[COUNT]]) : !hal.channel
  %channel = hal.channel.create device(%device : !hal.device)
                              affinity(%affinity)
                                 flags(0)
                                    id(%id)
                                 group(%group)
                                  rank(%rank)
                                 count(%count) : !hal.channel
  return
}

// -----

// CHECK-LABEL: @channel_split
//  CHECK-SAME: (%[[BASE_CHANNEL:.+]]: !hal.channel, %[[COLOR:.+]]: i32, %[[KEY:.+]]: i32)
func.func @channel_split(%base_channel: !hal.channel, %color: i32, %key: i32) {
  //      CHECK: %channel = hal.channel.split<%[[BASE_CHANNEL]] : !hal.channel>
  // CHECK-SAME:   color(%[[COLOR]])
  // CHECK-SAME:   key(%[[KEY]])
  // CHECK-SAME:   flags(0) : !hal.channel
  %channel = hal.channel.split<%base_channel : !hal.channel>
                              color(%color)
                                key(%key)
                              flags(0) : !hal.channel
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
