// RUN: iree-opt --split-input-file --iree-convert-hal-to-vm --canonicalize --iree-vm-target-index-bits=32 %s | FileCheck %s

// CHECK-LABEL: @channel_create
//  CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[AFFINITY:.+]]: i64, %[[ID:.+]]: !vm.buffer, %[[GROUP:.+]]: !vm.buffer, %[[RANK:.+]]: i32, %[[COUNT:.+]]: i32) -> !vm.ref<!hal.channel>
func.func @channel_create(%device: !hal.device, %affinity: i64, %id: !util.buffer, %group: !util.buffer, %rank: i32, %count: i32) -> !hal.channel {
  // CHECK: %[[FLAGS:.+]] = vm.const.i32.zero
  // CHECK: %[[CHANNEL:.+]] = vm.call @hal.channel.create(%[[DEVICE]], %[[AFFINITY]], %[[FLAGS]], %[[ID]], %[[GROUP]], %[[RANK]], %[[COUNT]])
  %channel = hal.channel.create device(%device : !hal.device)
                              affinity(%affinity)
                                 flags(0)
                                    id(%id)
                                 group(%group)
                                  rank(%rank)
                                 count(%count) : !hal.channel
  // CHECK: return %[[CHANNEL]]
  return %channel : !hal.channel
}

// -----

// CHECK-LABEL: @channel_split
//  CHECK-SAME: (%[[BASE_CHANNEL:.+]]: !vm.ref<!hal.channel>, %[[COLOR:.+]]: i32, %[[KEY:.+]]: i32)
func.func @channel_split(%base_channel: !hal.channel, %color: i32, %key: i32) -> !hal.channel {
  // CHECK: %[[FLAGS:.+]] = vm.const.i32.zero
  // CHECK: %[[SPLIT_CHANNEL:.+]] = vm.call @hal.channel.split(%[[BASE_CHANNEL]], %[[COLOR]], %[[KEY]], %[[FLAGS]])
  %split_channel = hal.channel.split<%base_channel : !hal.channel>
                               color(%color)
                                 key(%key)
                               flags(0) : !hal.channel
  // CHECK: return %[[SPLIT_CHANNEL]]
  return %split_channel : !hal.channel
}

// -----

// CHECK-LABEL: @channel_rank_and_count
//  CHECK-SAME: %[[CHANNEL:.+]]: !vm.ref<!hal.channel>
func.func @channel_rank_and_count(%channel: !hal.channel) -> (i32, i32) {
  // CHECK: %[[RANK_COUNT:.+]]:2 = vm.call @hal.channel.rank_and_count(%[[CHANNEL]])
  %rank, %count = hal.channel.rank_and_count<%channel : !hal.channel> : i32, i32
  // CHECK: return %[[RANK_COUNT]]#0, %[[RANK_COUNT]]#1
  return %rank, %count : i32, i32
}
