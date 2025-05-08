// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-vm-conversion{index-bits=32},canonicalize)' %s | FileCheck %s

// CHECK-LABEL: @channel_create
//  CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[AFFINITY:.+]]: i64, %[[ID:.+]]: !vm.buffer, %[[GROUP:.+]]: !vm.buffer, %[[RANK:.+]]: i32, %[[COUNT:.+]]: i32) -> !vm.ref<!hal.channel>
util.func public @channel_create(%device: !hal.device, %affinity: i64, %id: !util.buffer, %group: !util.buffer, %rank: i32, %count: i32) -> !hal.channel {
  // CHECK: %[[FLAGS:.+]] = vm.const.i64.zero
  // CHECK: %[[CHANNEL:.+]] = vm.call @hal.channel.create(%[[DEVICE]], %[[AFFINITY]], %[[FLAGS]], %[[ID]], %[[GROUP]], %[[RANK]], %[[COUNT]])
  %channel = hal.channel.create device(%device : !hal.device)
                              affinity(%affinity)
                                 flags("None")
                                    id(%id)
                                 group(%group)
                                  rank(%rank)
                                 count(%count) : !hal.channel
  // CHECK: vm.return %[[CHANNEL]]
  util.return %channel : !hal.channel
}

// -----

// CHECK-LABEL: @channel_split
//  CHECK-SAME: (%[[BASE_CHANNEL:.+]]: !vm.ref<!hal.channel>, %[[COLOR:.+]]: i32, %[[KEY:.+]]: i32)
util.func public @channel_split(%base_channel: !hal.channel, %color: i32, %key: i32) -> !hal.channel {
  // CHECK: %[[FLAGS:.+]] = vm.const.i64.zero
  // CHECK: %[[SPLIT_CHANNEL:.+]] = vm.call @hal.channel.split(%[[BASE_CHANNEL]], %[[COLOR]], %[[KEY]], %[[FLAGS]])
  %split_channel = hal.channel.split<%base_channel : !hal.channel>
                               color(%color)
                                 key(%key)
                               flags("None") : !hal.channel
  // CHECK: vm.return %[[SPLIT_CHANNEL]]
  util.return %split_channel : !hal.channel
}

// -----

// CHECK-LABEL: @channel_rank_and_count
//  CHECK-SAME: %[[CHANNEL:.+]]: !vm.ref<!hal.channel>
util.func public @channel_rank_and_count(%channel: !hal.channel) -> (i32, i32) {
  // CHECK: %[[RANK_COUNT:.+]]:2 = vm.call @hal.channel.rank_and_count(%[[CHANNEL]])
  %rank, %count = hal.channel.rank_and_count<%channel : !hal.channel> : i32, i32
  // CHECK: vm.return %[[RANK_COUNT]]#0, %[[RANK_COUNT]]#1
  util.return %rank, %count : i32, i32
}
