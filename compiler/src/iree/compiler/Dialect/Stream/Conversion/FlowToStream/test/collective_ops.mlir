// RUN: iree-opt --split-input-file --iree-stream-conversion %s | FileCheck %s

// CHECK-LABEL: @channel_count
func.func @channel_count() -> index {
  // CHECK: [[CHANNEL:%.+]] = stream.channel.default : !stream.channel
  // CHECK: [[COUNT:%.+]] = stream.channel.count [[CHANNEL]] : index
  // CHECK: return [[COUNT]] : index
  %channel_default = flow.channel.default : !flow.channel
  %count = flow.channel.count %channel_default : index
  return %count : index
}

//-----

// CHECK-LABEL: @channel_rank
func.func @channel_rank() -> index {
  // CHECK: [[CHANNEL:%.+]] = stream.channel.default : !stream.channel
  // CHECK: [[RANK:%.+]] = stream.channel.rank [[CHANNEL]] : index
  // CHECK: return [[RANK]] : index
  %channel_default = flow.channel.default : !flow.channel
  %rank = flow.channel.rank %channel_default : index
  return %rank : index
}
