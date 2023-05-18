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

// -----

// CHECK-LABEL: @channel_count
//  CHECK-SAME: (%[[CHANNEL:.+]]: !stream.channel)
func.func @channel_count(%channel: !flow.channel) -> index {
  // CHECK: %[[COUNT:.+]] = stream.channel.count %[[CHANNEL]] : index
  // CHECK: return %[[COUNT]] : index
  %count = flow.channel.count %channel : index
  return %count : index
}

// -----

// CHECK-LABEL: @all_reduce_sum
func.func @all_reduce_sum(%channel: !flow.channel, %arg0: tensor<2304xf32>) -> tensor<2304xf32> {
  // CHECK: stream.tensor.empty : tensor<2304xf32>
  // CHECK: stream.async.collective<all_reduce with sum : f32>
  %0 = flow.tensor.empty : tensor<2304xf32>
  %1 = flow.collective.all_reduce sum, f32, %0, %arg0, %channel : (tensor<2304xf32>, tensor<2304xf32>, !flow.channel) -> tensor<2304xf32>
  return %1 : tensor<2304xf32>
}

// -----

// CHECK-LABEL: @all_gather
func.func @all_gather(%channel: !flow.channel, %arg0: tensor<512xf32>) -> tensor<1024xf32> {
  // CHECK: stream.tensor.empty : tensor<1024xf32>
  // CHECK: stream.async.collective<all_gather : f32>
  %0 = flow.tensor.empty : tensor<1024xf32>
  %1 = flow.collective.all_gather f32, %0, %arg0, %channel : (tensor<1024xf32>, tensor<512xf32>, !flow.channel) -> tensor<1024xf32>
  return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: @all_to_all
func.func @all_to_all(%channel: !flow.channel, %arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: stream.tensor.empty : tensor<1024xf32>
  // CHECK: stream.async.collective<all_to_all : f32>
  %0 = flow.tensor.empty : tensor<1024xf32>
  %1 = flow.collective.all_to_all f32, %0, %arg0, %channel : (tensor<1024xf32>, tensor<1024xf32>, !flow.channel) -> tensor<1024xf32>
  return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: @reduce_scatter
func.func @reduce_scatter(%channel: !flow.channel, %arg0: tensor<4x2xf32>) -> tensor<2x2xf32> {
  // CHECK: stream.tensor.empty : tensor<2x2xf32>
  // CHECK: stream.async.collective<reduce_scatter with sum : f32>
  %0 = flow.tensor.empty : tensor<2x2xf32>
  %1 = flow.collective.reduce_scatter sum, f32, %0, %arg0, %channel : (tensor<2x2xf32>, tensor<4x2xf32>, !flow.channel) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: @send_recv
// CHECK-SAME: index, %[[SEND:.+]]: index, %[[RECV:.+]]: index)
func.func @send_recv(%channel: !flow.channel, %arg0: tensor<1024xf32>, %send: index, %recv: index) -> tensor<1024xf32> {
  // CHECK: stream.tensor.empty : tensor<1024xf32>
  // CHECK-DAG: %[[CST_LO_MASK:.+]] = arith.constant 65535 : i32
  // CHECK-DAG: %[[CST_SHIFT16:.+]] = arith.constant 16 : i32
  // CHECK-DAG: %[[SEND_I32:.+]]  = arith.index_cast %[[SEND]] : index to i32
  // CHECK-DAG: %[[RECV_I32:.+]]  = arith.index_cast %[[RECV]] : index to i32
  // CHECK-DAG: %[[LO:.+]] = arith.andi %[[SEND_I32]], %[[CST_LO_MASK]] : i32
  // CHECK-DAG: %[[HI:.+]] = arith.shli %[[RECV_I32]], %[[CST_SHIFT16]] : i32
  // CHECK-DAG: %[[PARAM:.+]] = arith.ori %[[HI]], %[[LO]] : i32
  // CHECK: stream.async.collective<send_recv : f32>
  // CHECK-SAME: source_target_pair(%[[PARAM]])
  %0 = flow.tensor.empty : tensor<1024xf32>
  %1 = flow.collective.send_recv f32, %0, %arg0, %channel, %send, %recv : (tensor<1024xf32>, tensor<1024xf32>, !flow.channel, index, index) -> tensor<1024xf32>
  return %1 : tensor<1024xf32>
}
