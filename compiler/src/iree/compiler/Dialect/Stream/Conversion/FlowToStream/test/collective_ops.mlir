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

//-----

// CHECK-LABEL: @all_reduce_sum
func.func @all_reduce_sum(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  // CHECK: stream.channel.default
  // CHECK: stream.tensor.empty : tensor<2304xf32>
  // CHECK: stream.async.collective<all_reduce with sum : f32>
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<2304xf32>
  %channel_default = flow.channel.default : !flow.channel
  %1 = flow.tensor.empty : tensor<2304xf32>
  %2 = flow.collective.all_reduce sum, f32, %1, %0, %channel_default : (tensor<2304xf32>, tensor<2304xf32>, !flow.channel) -> tensor<2304xf32>
  %3 = hal.tensor.export %2 : tensor<2304xf32> -> !hal.buffer_view
  return %3 : !hal.buffer_view
}

//-----

// CHECK-LABEL: @allgather
func.func @allgather(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  // CHECK: stream.channel.default
  // CHECK: stream.tensor.empty : tensor<1024xf32>
  // CHECK: stream.async.collective<all_gather : f32>
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<512xf32>
  %channel_default = flow.channel.default : !flow.channel
  %1 = flow.tensor.empty : tensor<1024xf32>
  %2 = flow.collective.all_gather f32, %1, %0, %channel_default : (tensor<1024xf32>, tensor<512xf32>, !flow.channel) -> tensor<1024xf32>
  %3 = hal.tensor.export %2 : tensor<1024xf32> -> !hal.buffer_view
  return %3 : !hal.buffer_view
}

//-----

// CHECK-LABEL: @all_to_all
func.func @all_to_all(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  // CHECK: stream.channel.default
  // CHECK: stream.tensor.empty : tensor<1024xf32>
  // CHECK: stream.async.collective<all_to_all : f32>
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<1024xf32>
  %channel_default = flow.channel.default : !flow.channel
  %1 = flow.tensor.empty : tensor<1024xf32>
  %2 = flow.collective.all_to_all f32, %1, %0, %channel_default : (tensor<1024xf32>, tensor<1024xf32>, !flow.channel) -> tensor<1024xf32>
  %3 = hal.tensor.export %2 : tensor<1024xf32> -> !hal.buffer_view
  return %3 : !hal.buffer_view
}

//-----

// CHECK-LABEL: @reduce_scatter
func.func @reduce_scatter(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  // CHECK: stream.channel.default
  // CHECK: stream.tensor.empty : tensor<2x2xf32>
  // CHECK: stream.async.collective<reduce_scatter with sum : f32>
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<4x2xf32>
  %channel_default = flow.channel.default : !flow.channel
  %1 = flow.tensor.empty : tensor<2x2xf32>
  %2 = flow.collective.reduce_scatter sum, f32, %1, %0, %channel_default : (tensor<2x2xf32>, tensor<4x2xf32>, !flow.channel) -> tensor<2x2xf32>
  %3 = hal.tensor.export %2 : tensor<2x2xf32> -> !hal.buffer_view
  return %3 : !hal.buffer_view
}
