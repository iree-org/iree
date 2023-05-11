// RUN: iree-opt --split-input-file --iree-stream-conversion %s | FileCheck %s

// CHECK-LABEL: @channel_split
//  CHECK-SAME: (%[[BASE_CHANNEL:.+]]: !stream.channel)
func.func @channel_split(%base_channel: !flow.channel) -> !flow.channel {
  // CHECK-DAG: %[[COLOR:.+]] = arith.constant 100 : index
  %color = arith.constant 100 : index
  // CHECK-DAG: %[[KEY:.+]] = arith.constant 101 : index
  %key = arith.constant 101 : index
  // CHECK: %[[SPLIT_CHANNEL:.+]] = stream.channel.split %[[BASE_CHANNEL]], %[[COLOR]], %[[KEY]] : !stream.channel -> !stream.channel
  %split_channel = flow.channel.split %base_channel, %color, %key : !flow.channel -> !flow.channel
  // CHECK: return %[[SPLIT_CHANNEL]]
  return %split_channel : !flow.channel
}

// -----

// CHECK-LABEL: @channel_rank
//  CHECK-SAME: (%[[CHANNEL:.+]]: !stream.channel)
func.func @channel_rank(%channel: !flow.channel) -> index {
  // CHECK: %[[RANK:.+]] = stream.channel.rank %[[CHANNEL]] : index
  // CHECK: return %[[RANK]] : index
  %rank = flow.channel.rank %channel : index
  return %rank : index
}

//-----

// CHECK-LABEL: @channel_count
//  CHECK-SAME: (%[[CHANNEL:.+]]: !stream.channel)
func.func @channel_count(%channel: !flow.channel) -> index {
  // CHECK: %[[COUNT:.+]] = stream.channel.count %[[CHANNEL]] : index
  // CHECK: return %[[COUNT]] : index
  %count = flow.channel.count %channel : index
  return %count : index
}

//-----

// CHECK-LABEL: @all_reduce_sum
func.func @all_reduce_sum(%channel: !flow.channel, %arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  // CHECK: stream.tensor.empty : tensor<2304xf32>
  // CHECK: stream.async.collective<all_reduce with sum : f32>
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<2304xf32>
  %1 = flow.tensor.empty : tensor<2304xf32>
  %2 = flow.collective.all_reduce sum, f32, %1, %0, %channel : (tensor<2304xf32>, tensor<2304xf32>, !flow.channel) -> tensor<2304xf32>
  %3 = hal.tensor.export %2 : tensor<2304xf32> -> !hal.buffer_view
  return %3 : !hal.buffer_view
}

//-----

// CHECK-LABEL: @all_gather
func.func @all_gather(%channel: !flow.channel, %arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  // CHECK: stream.tensor.empty : tensor<1024xf32>
  // CHECK: stream.async.collective<all_gather : f32>
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<512xf32>
  %1 = flow.tensor.empty : tensor<1024xf32>
  %2 = flow.collective.all_gather f32, %1, %0, %channel : (tensor<1024xf32>, tensor<512xf32>, !flow.channel) -> tensor<1024xf32>
  %3 = hal.tensor.export %2 : tensor<1024xf32> -> !hal.buffer_view
  return %3 : !hal.buffer_view
}

//-----

// CHECK-LABEL: @all_to_all
func.func @all_to_all(%channel: !flow.channel, %arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  // CHECK: stream.tensor.empty : tensor<1024xf32>
  // CHECK: stream.async.collective<all_to_all : f32>
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<1024xf32>
  %1 = flow.tensor.empty : tensor<1024xf32>
  %2 = flow.collective.all_to_all f32, %1, %0, %channel : (tensor<1024xf32>, tensor<1024xf32>, !flow.channel) -> tensor<1024xf32>
  %3 = hal.tensor.export %2 : tensor<1024xf32> -> !hal.buffer_view
  return %3 : !hal.buffer_view
}

//-----

// CHECK-LABEL: @reduce_scatter
func.func @reduce_scatter(%channel: !flow.channel, %arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  // CHECK: stream.tensor.empty : tensor<2x2xf32>
  // CHECK: stream.async.collective<reduce_scatter with sum : f32>
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<4x2xf32>
  %1 = flow.tensor.empty : tensor<2x2xf32>
  %2 = flow.collective.reduce_scatter sum, f32, %1, %0, %channel : (tensor<2x2xf32>, tensor<4x2xf32>, !flow.channel) -> tensor<2x2xf32>
  %3 = hal.tensor.export %2 : tensor<2x2xf32> -> !hal.buffer_view
  return %3 : !hal.buffer_view
}
