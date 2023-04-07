// RUN: iree-opt --split-input-file --iree-convert-hal-to-vm --canonicalize --iree-vm-target-index-bits=32 %s | FileCheck %s

// CHECK-LABEL: @channel_default
//  CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[AFFINITY:.+]]: i64, %[[ID:.+]]: !vm.buffer) -> !vm.ref<!hal.channel>
func.func @channel_default(%device: !hal.device, %affinity: i64, %id: !util.buffer) -> !hal.channel {
  // CHECK: %[[FLAGS:.+]] = vm.const.i32.zero
  // CHECK: %[[CHANNEL:.+]] = vm.call @hal.channel.default(%[[DEVICE]], %[[AFFINITY]], %[[FLAGS]], %[[ID]]
  %channel = hal.channel.default device(%device : !hal.device)
                               affinity(%affinity)
                                  flags(0)
                                     id(%id) : !hal.channel
  // CHECK: return %[[CHANNEL]]
  return %channel : !hal.channel
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
