// RUN: iree-opt --split-input-file --iree-mhlo-to-linalg-on-tensors --canonicalize -cse %s | FileCheck %s

// CHECK-LABEL: @replica_id
func.func @replica_id() -> tensor<ui32> {
  // CHECK-DAG: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK-DAG: %[[RANK:.+]] = flow.channel.rank %[[CHANNEL]] : index
  // CHECK-DAG: %[[CAST:.+]] = arith.index_castui %[[RANK]] : index to i32
  // CHECK-DAG: %[[TENSOR:.+]] = tensor.from_elements %[[CAST]] : tensor<i32>
  // CHECK-DAG: return %[[TENSOR]] : tensor<i32>
  %id = mhlo.replica_id : tensor<ui32>
  return %id : tensor<ui32>
}

// -----

module @jit_fn attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 4 : i32 } {
  // CHECK-LABEL: @replica_id_with_partitions
  func.func @replica_id_with_partitions() -> tensor<ui32> {
    // CHECK-DAG: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
    // CHECK-DAG: %[[RANK:.+]] = flow.channel.rank %[[CHANNEL]] : index
    // CHECK-DAG: %[[DIV2:.+]] = arith.divui %[[RANK]], %c2 : index
    // CHECK-DAG: %[[CAST:.+]] = arith.index_castui %[[DIV2]] : index to i32
    // CHECK-DAG: %[[TENSOR:.+]] = tensor.from_elements %[[CAST]] : tensor<i32>
    // CHECK-DAG: return %[[TENSOR]] : tensor<i32>
    %id = mhlo.replica_id : tensor<ui32>
    return %id : tensor<ui32>
  }
}

// -----

// Returns 0 since num_partitions is not set.

// CHECK-LABEL: @partition_id
func.func @partition_id() -> tensor<ui32> {
  // CHECK-DAG: %[[CST0:.+]] = arith.constant dense<0> : tensor<i32>
  // CHECK-DAG: return %[[CST0]] : tensor<i32>
  %id = mhlo.partition_id : tensor<ui32>
  return %id : tensor<ui32>
}

// -----

module @jit_fn attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 4 : i32 } {
  // CHECK-LABEL: @partition_id_with_partitions
  func.func @partition_id_with_partitions() -> tensor<ui32> {
    // CHECK-DAG: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
    // CHECK-DAG: %[[RANK:.+]] = flow.channel.rank %[[CHANNEL]] : index
    // CHECK-DAG: %[[REM2:.+]] = arith.remui %[[RANK]], %c2 : index
    // CHECK-DAG: %[[CAST:.+]] = arith.index_castui %[[REM2]] : index to i32
    // CHECK-DAG: %[[TENSOR:.+]] = tensor.from_elements %[[CAST]] : tensor<i32>
    // CHECK-DAG: return %[[TENSOR]] : tensor<i32>
    %id = mhlo.partition_id : tensor<ui32>
    return %id : tensor<ui32>
  }
}

// -----

// CHECK-LABEL: @all_reduce_sum
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2304xf32>)
func.func @all_reduce_sum(%input : tensor<2304xf32>) -> tensor<2304xf32> {
  // CHECK: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<2304xf32>
  // CHECK: %[[OP:.+]] = flow.collective.all_reduce sum, f32, %[[EMPTY]], %[[ARG0]], %[[CHANNEL]]  : (tensor<2304xf32>, tensor<2304xf32>, !flow.channel) -> %[[EMPTY]] as tensor<2304xf32>
  // CHECK: return %[[OP]] : tensor<2304xf32>
  %out = "mhlo.all_reduce"(%input) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %sum = mhlo.add %arg0, %arg1 : tensor<f32>
      mhlo.return %sum : tensor<f32>
    }) {channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
        replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>,
        use_global_device_ids} : (tensor<2304xf32>) -> tensor<2304xf32>
  return %out : tensor<2304xf32>
}

// -----

// CHECK-LABEL: @all_reduce_product
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2304xf32>)
func.func @all_reduce_product(%input : tensor<2304xf32>) -> tensor<2304xf32> {
  // CHECK: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<2304xf32>
  // CHECK: %[[OP:.+]] = flow.collective.all_reduce product, f32, %[[EMPTY]], %[[ARG0]], %[[CHANNEL]]  : (tensor<2304xf32>, tensor<2304xf32>, !flow.channel) -> %[[EMPTY]] as tensor<2304xf32>
  // CHECK: return %[[OP]] : tensor<2304xf32>
  %out = "mhlo.all_reduce"(%input) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %mul = mhlo.multiply %arg0, %arg1 : tensor<f32>
      mhlo.return %mul : tensor<f32>
    }) {channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
        replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>,
        use_global_device_ids} : (tensor<2304xf32>) -> tensor<2304xf32>
  return %out : tensor<2304xf32>
}

// -----

// CHECK-LABEL: @all_reduce_minimum
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2304xf32>)
func.func @all_reduce_minimum(%input : tensor<2304xf32>) -> tensor<2304xf32> {
  // CHECK: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<2304xf32>
  // CHECK: %[[OP:.+]] = flow.collective.all_reduce minimum, f32, %[[EMPTY]], %[[ARG0]], %[[CHANNEL]]  : (tensor<2304xf32>, tensor<2304xf32>, !flow.channel) -> %[[EMPTY]] as tensor<2304xf32>
  // CHECK: return %[[OP]] : tensor<2304xf32>
  %out = "mhlo.all_reduce"(%input) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %mul = mhlo.minimum %arg0, %arg1 : tensor<f32>
      mhlo.return %mul : tensor<f32>
    }) {channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
        replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>,
        use_global_device_ids} : (tensor<2304xf32>) -> tensor<2304xf32>
  return %out : tensor<2304xf32>
}

// -----

// CHECK-LABEL: @all_reduce_maximum
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2304xf32>)
func.func @all_reduce_maximum(%input : tensor<2304xf32>) -> tensor<2304xf32> {
  // CHECK: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<2304xf32>
  // CHECK: %[[OP:.+]] = flow.collective.all_reduce maximum, f32, %[[EMPTY]], %[[ARG0]], %[[CHANNEL]]  : (tensor<2304xf32>, tensor<2304xf32>, !flow.channel) -> %[[EMPTY]] as tensor<2304xf32>
  // CHECK: return %[[OP]] : tensor<2304xf32>
  %out = "mhlo.all_reduce"(%input) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %mul = mhlo.maximum %arg0, %arg1 : tensor<f32>
      mhlo.return %mul : tensor<f32>
    }) {channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
        replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>,
        use_global_device_ids} : (tensor<2304xf32>) -> tensor<2304xf32>
  return %out : tensor<2304xf32>
}

// -----

// CHECK-LABEL: @all_reduce_maximum_optional_attrs
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2304xf32>)
func.func @all_reduce_maximum_optional_attrs(%input : tensor<2304xf32>) -> tensor<2304xf32> {
  // CHECK: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<2304xf32>
  // CHECK: %[[OP:.+]] = flow.collective.all_reduce maximum, f32, %[[EMPTY]], %[[ARG0]], %[[CHANNEL]]  : (tensor<2304xf32>, tensor<2304xf32>, !flow.channel) -> %[[EMPTY]] as tensor<2304xf32>
  // CHECK: return %[[OP]] : tensor<2304xf32>
  %out = "mhlo.all_reduce"(%input) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %mul = mhlo.maximum %arg0, %arg1 : tensor<f32>
      mhlo.return %mul : tensor<f32>
    }) {replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>} : (tensor<2304xf32>) -> tensor<2304xf32>
  return %out : tensor<2304xf32>
}

// -----

// CHECK-LABEL: @all_reduce_sum_with_groups
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2x4xi32>)
func.func @all_reduce_sum_with_groups(%input : tensor<2x4xi32>) -> tensor<2x4xi32> {
  // CHECK: %[[BASE_CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: %[[BASE_RANK:.+]] = flow.channel.rank %[[BASE_CHANNEL]]
  // CHECK: %[[SPLIT_COLOR:.+]] = util.switch index from [%c0, %c1] at %[[BASE_RANK]] else %c-1
  // CHECK: %[[SPLIT_KEY:.+]] = util.switch index from [%c0, %c0] at %[[BASE_RANK]] else %c-1
  // CHECK: %[[SPLIT_CHANNEL:.+]] = flow.channel.split %[[BASE_CHANNEL]], %[[SPLIT_COLOR]], %[[SPLIT_KEY]] : !flow.channel
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<2x4xi32>
  // CHECK: %[[OP:.+]] = flow.collective.all_reduce sum, ui32, %[[EMPTY]], %[[ARG0]], %[[SPLIT_CHANNEL]] : (tensor<2x4xi32>, tensor<2x4xi32>, !flow.channel) -> %[[EMPTY]] as tensor<2x4xi32>
  // CHECK: return %[[OP]] : tensor<2x4xi32>
  %out = "mhlo.all_reduce"(%input) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %sum = mhlo.add %arg0, %arg1 : tensor<i32>
      mhlo.return %sum : tensor<i32>
    }) {channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
        replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
        use_global_device_ids} : (tensor<2x4xi32>) -> tensor<2x4xi32>
  return %out : tensor<2x4xi32>
}

// -----

// CHECK-LABEL: @all_gather_dim_0
// CHECK-SAME: (%[[ARG0:.+]]: tensor<512xf32>) -> tensor<1024xf32>
func.func @all_gather_dim_0(%input : tensor<512xf32>) -> tensor<1024xf32> {
  // CHECK: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1024xf32>
  // CHECK: %[[OP:.+]] = flow.collective.all_gather f32, %[[EMPTY]], %[[ARG0]], %[[CHANNEL]]  : (tensor<1024xf32>, tensor<512xf32>, !flow.channel) -> %[[EMPTY]] as tensor<1024xf32>
  // CHECK: return %[[OP]] : tensor<1024xf32>
  %out = "mhlo.all_gather"(%input) {all_gather_dim = 0 : i64,
     channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
     replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
     use_global_device_ids} : (tensor<512xf32>) -> tensor<1024xf32>
  return %out : tensor<1024xf32>
}

// -----

// CHECK-LABEL: @all_gather_dim_1
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2x2xf32>) -> tensor<2x4xf32>
func.func @all_gather_dim_1(%input : tensor<2x2xf32>) -> tensor<2x4xf32> {
  // CHECK: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: tensor.empty() : tensor<2x2xf32>
  // CHECK: %[[TRANSPOSE_ARG:.+]] = linalg.generic
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<4x2xf32>
  // CHECK: %[[OP:.+]] = flow.collective.all_gather f32, %[[EMPTY]], %[[TRANSPOSE_ARG]], %[[CHANNEL]]  : (tensor<4x2xf32>, tensor<2x2xf32>, !flow.channel) -> %[[EMPTY]] as tensor<4x2xf32>
  // CHECK: tensor.empty() : tensor<2x4xf32>
  // CHECK: %[[TRANSPOSE_OUT:.+]] = linalg.generic
  // CHECK: return %[[TRANSPOSE_OUT]] : tensor<2x4xf32>
  %out = "mhlo.all_gather"(%input) {all_gather_dim = 1 : i64,
     channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
     replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
     use_global_device_ids} : (tensor<2x2xf32>) -> tensor<2x4xf32>
  return %out : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: @all_gather_dim_0_optional_attrs
// CHECK-SAME: (%[[ARG0:.+]]: tensor<512xf32>) -> tensor<1024xf32>
func.func @all_gather_dim_0_optional_attrs(%input : tensor<512xf32>) -> tensor<1024xf32> {
  // CHECK: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1024xf32>
  // CHECK: %[[OP:.+]] = flow.collective.all_gather f32, %[[EMPTY]], %[[ARG0]], %[[CHANNEL]]  : (tensor<1024xf32>, tensor<512xf32>, !flow.channel) -> %[[EMPTY]] as tensor<1024xf32>
  // CHECK: return %[[OP]] : tensor<1024xf32>
  %out = "mhlo.all_gather"(%input) {all_gather_dim = 0 : i64,
     replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<512xf32>) -> tensor<1024xf32>
  return %out : tensor<1024xf32>
}

// -----

// CHECK-LABEL: @all_to_all_split_concat_same
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1024xf32>) -> tensor<1024xf32>
func.func @all_to_all_split_concat_same(%input : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1024xf32>
  // CHECK: %[[OP:.+]] = flow.collective.all_to_all f32, %[[EMPTY]], %[[ARG0]], %[[CHANNEL]]  : (tensor<1024xf32>, tensor<1024xf32>, !flow.channel) -> %[[EMPTY]] as tensor<1024xf32>
  // CHECK: return %[[OP]] : tensor<1024xf32>
  %out = "mhlo.all_to_all"(%input) {
     split_dimension = 0 : i64,
     concat_dimension = 0 : i64,
     split_count = 2 : i64,
     channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
     replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<1024xf32>) -> tensor<1024xf32>
  return %out : tensor<1024xf32>
}

// -----

// CHECK-LABEL: @all_to_all_split_concat_same_dim_1
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2x4xf32>) -> tensor<2x4xf32>
func.func @all_to_all_split_concat_same_dim_1(%input : tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<4x2xf32>
  // CHECK: %[[TRANSPOSE_ARG:.+]] = linalg.generic
  // CHECK: %[[OP:.+]] = flow.collective.all_to_all f32, %[[EMPTY]], %[[TRANSPOSE_ARG]], %[[CHANNEL]]  : (tensor<4x2xf32>, tensor<4x2xf32>, !flow.channel) -> %[[EMPTY]] as tensor<4x2xf32>
  // CHECK: %[[TRANSPOSE_OUT:.+]] = linalg.generic
  // CHECK: return %[[TRANSPOSE_OUT]] : tensor<2x4xf32>
  %out = "mhlo.all_to_all"(%input) {
     split_dimension = 1 : i64,
     concat_dimension = 1 : i64,
     split_count = 2 : i64,
     channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
     replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %out : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: @all_to_all_split_dim_0
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x4xf32>) -> tensor<2x8xf32>
func.func @all_to_all_split_dim_0(%input : tensor<4x4xf32>) -> tensor<2x8xf32> {
  // CHECK: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<4x4xf32>
  // CHECK: %[[OP:.+]] = flow.collective.all_to_all f32, %[[EMPTY]], %[[ARG0]], %[[CHANNEL]]  : (tensor<4x4xf32>, tensor<4x4xf32>, !flow.channel) -> %[[EMPTY]] as tensor<4x4xf32>
  // CHECK: %[[REARRANGE_RESHAPE:.+]] = tensor.expand_shape %[[OP]] {{\[}}[0, 1], [2]] : tensor<4x4xf32> into tensor<2x2x4xf32>
  // CHECK: %[[REARRANGE_TRANSPOSE:.+]] = linalg.generic
  // CHECK: %[[RESHAPE_OUT:.+]] = tensor.collapse_shape %[[REARRANGE_TRANSPOSE]] {{\[}}[0], [1, 2]] : tensor<2x2x4xf32> into tensor<2x8xf32>
  // CHECK: return %[[RESHAPE_OUT]] : tensor<2x8xf32>
  %out = "mhlo.all_to_all"(%input) {
     split_dimension = 0 : i64,
     concat_dimension = 1 : i64,
     split_count = 2 : i64,
     channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
     replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<4x4xf32>) -> tensor<2x8xf32>
  return %out : tensor<2x8xf32>
}

// -----

// CHECK-LABEL: @all_to_all_split_dim_1
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x4xf32>) -> tensor<8x2xf32>
func.func @all_to_all_split_dim_1(%input : tensor<4x4xf32>) -> tensor<8x2xf32> {
  // CHECK: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<4x4xf32>
  // CHECK: %[[TRANSPOSE_ARG:.+]] = linalg.generic
  // CHECK: %[[OP:.+]] = flow.collective.all_to_all f32, %[[EMPTY]], %[[TRANSPOSE_ARG]], %[[CHANNEL]]  : (tensor<4x4xf32>, tensor<4x4xf32>, !flow.channel) -> %[[EMPTY]] as tensor<4x4xf32>
  // CHECK: %[[TRANSPOSE_OUT:.+]] = linalg.generic
  // CHECK: %[[REARRANGE_RESHAPE1:.+]] = tensor.expand_shape %[[TRANSPOSE_OUT]] {{\[}}[0], [1, 2]] : tensor<4x4xf32> into tensor<4x2x2xf32>
  // CHECK: %[[EMPTY2:.+]] = tensor.empty() : tensor<2x4x2xf32>
  // CHECK: %[[REARRANGE_TRANSPOSE:.+]] = linalg.generic
  // CHECK: %[[REARRANGE_RESHAPE2:.+]] = tensor.collapse_shape %[[REARRANGE_TRANSPOSE]] {{\[}}[0, 1], [2]] : tensor<2x4x2xf32> into tensor<8x2xf32>
  // CHECK: return %[[REARRANGE_RESHAPE2]] : tensor<8x2xf32>
  %out = "mhlo.all_to_all"(%input) {
     split_dimension = 1 : i64,
     concat_dimension = 0 : i64,
     split_count = 2 : i64,
     channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
     replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<4x4xf32>) -> tensor<8x2xf32>
  return %out : tensor<8x2xf32>
}

// -----

// CHECK-LABEL: @all_to_all_3d_split_dim_1
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x4x4xf32>) -> tensor<4x2x8xf32>
func.func @all_to_all_3d_split_dim_1(%input : tensor<4x4x4xf32>) -> tensor<4x2x8xf32> {
  // CHECK: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<4x4x4xf32>
  // CHECK: %[[TRANSPOSE_ARG:.+]] = linalg.generic
  // CHECK: %[[OP:.+]] = flow.collective.all_to_all f32, %[[EMPTY]], %[[TRANSPOSE_ARG]], %[[CHANNEL]]  : (tensor<4x4x4xf32>, tensor<4x4x4xf32>, !flow.channel) -> %[[EMPTY]] as tensor<4x4x4xf32>
  // CHECK: %[[TRANSPOSE_OUT:.+]] = linalg.generic
  // CHECK: %[[REARRANGE_RESHAPE1:.+]] = tensor.expand_shape %[[TRANSPOSE_OUT]] {{\[}}[0], [1, 2], [3]] : tensor<4x4x4xf32> into tensor<4x2x2x4xf32>
  // CHECK: %[[EMPTY_1:.+]] = tensor.empty() : tensor<4x2x2x4xf32>
  // CHECK: %[[REARRANGE_TRANSPOSE:.+]] = linalg.generic
  // CHECK: %[[REARRANGE_RESHAPE2:.+]] = tensor.collapse_shape %[[REARRANGE_TRANSPOSE]] {{\[}}[0], [1], [2, 3]] : tensor<4x2x2x4xf32> into tensor<4x2x8xf32>
  // CHECK: return %[[REARRANGE_RESHAPE2]] : tensor<4x2x8xf32>
  %out = "mhlo.all_to_all"(%input) {
     split_dimension = 1 : i64,
     concat_dimension = 2 : i64,
     split_count = 2 : i64,
     channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
     replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<4x4x4xf32>) -> tensor<4x2x8xf32>
  return %out : tensor<4x2x8xf32>
}

// -----

// CHECK-LABEL: @reduce_scatter_dim_0
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x2xf32>) -> tensor<2x2xf32>
func.func @reduce_scatter_dim_0(%input : tensor<4x2xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<2x2xf32>
  // CHECK: %[[OP:.+]] = flow.collective.reduce_scatter sum, f32, %[[EMPTY]], %[[ARG0]], %[[CHANNEL]]  : (tensor<2x2xf32>, tensor<4x2xf32>, !flow.channel) -> %[[EMPTY]] as tensor<2x2xf32>
  // CHECK: return %[[OP]] : tensor<2x2xf32>
  %out = "mhlo.reduce_scatter"(%input) ({
  ^bb0(%arg0: tensor<f32> , %arg1: tensor<f32>) :
    %sum = mhlo.add %arg0, %arg1 : tensor<f32>
    mhlo.return %sum : tensor<f32>
  }) {scatter_dimension = 0 : i64,
      channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      use_global_device_ids} : (tensor<4x2xf32>) -> tensor<2x2xf32>
  return %out : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: @reduce_scatter_dim_1
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2x4xf32>) -> tensor<2x2xf32>
func.func @reduce_scatter_dim_1(%input : tensor<2x4xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: tensor.empty() : tensor<4x2xf32>
  // CHECK: %[[TRANSPOSE_ARG:.+]] = linalg.generic
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<2x2xf32>
  // CHECK: %[[OP:.+]] = flow.collective.reduce_scatter sum, f32, %[[EMPTY]], %[[TRANSPOSE_ARG]], %[[CHANNEL]]  : (tensor<2x2xf32>, tensor<4x2xf32>, !flow.channel) -> %[[EMPTY]] as tensor<2x2xf32>
  // CHECK: %[[TRANSPOSE_OUT:.+]] = linalg.generic
  // CHECK: return %[[TRANSPOSE_OUT]] : tensor<2x2xf32>
  %out = "mhlo.reduce_scatter"(%input) ({
  ^bb0(%arg0: tensor<f32> , %arg1: tensor<f32>) :
    %sum = mhlo.add %arg0, %arg1 : tensor<f32>
    mhlo.return %sum : tensor<f32>
  }) {scatter_dimension = 1 : i64,
      channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      use_global_device_ids} : (tensor<2x4xf32>) -> tensor<2x2xf32>
  return %out : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: @reduce_scatter_dim_0_optional_attrs
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x2xf32>) -> tensor<2x2xf32>
func.func @reduce_scatter_dim_0_optional_attrs(%input : tensor<4x2xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<2x2xf32>
  // CHECK: %[[OP:.+]] = flow.collective.reduce_scatter sum, f32, %[[EMPTY]], %[[ARG0]], %[[CHANNEL]]  : (tensor<2x2xf32>, tensor<4x2xf32>, !flow.channel) -> %[[EMPTY]] as tensor<2x2xf32>
  // CHECK: return %[[OP]] : tensor<2x2xf32>
  %out = "mhlo.reduce_scatter"(%input) ({
  ^bb0(%arg0: tensor<f32> , %arg1: tensor<f32>) :
    %sum = mhlo.add %arg0, %arg1 : tensor<f32>
    mhlo.return %sum : tensor<f32>
  }) {scatter_dimension = 0 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<4x2xf32>) -> tensor<2x2xf32>
  return %out : tensor<2x2xf32>
}

// -----

// flattened_ids: channel_id > 0 && use_global_device_ids = true
module @jit_fn attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 8 : i32 } {
  // CHECK-LABEL: @flattened_ids
  // CHECK-SAME: ([[ARG0:%.+]]: tensor<2304xf32>)
  func.func @flattened_ids(%input : tensor<2304xf32>) -> tensor<2304xf32> {
    // CHECK: [[CHANNEL:%.+]] = flow.channel.default : !flow.channel
    // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<2304xf32>
    // CHECK: [[ALLREDUCE:%.+]] = flow.collective.all_reduce sum, f32, [[EMPTY]], [[ARG0]], [[CHANNEL]] : (tensor<2304xf32>, tensor<2304xf32>, !flow.channel) -> [[EMPTY]] as tensor<2304xf32>
    // CHECK: return [[ALLREDUCE]] : tensor<2304xf32>
    %out = "mhlo.all_reduce"(%input) ({
      ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
        %sum = mhlo.add %arg0, %arg1 : tensor<f32>
        mhlo.return %sum : tensor<f32>
      }) {channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
          replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>,
          use_global_device_ids} : (tensor<2304xf32>) -> tensor<2304xf32>
    return %out : tensor<2304xf32>
  }
}

// -----

// cross-replica: channel_id <= 0 && use_global_device_ids = false
module @jit_fn attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 4 : i32 } {
  // CHECK-LABEL: @cross_replica
  func.func @cross_replica(%input : tensor<2304xf32>) -> tensor<2304xf32> {
    // Cross replica should form groups (0,2,4,6),(1,3,5,7), where each number represents a cell below.
    // +---+---+
    // | 0 | 1 |
    // | 2 | 3 |
    // | 4 | 5 |
    // | 6 | 7 |
    // +---+---+
    //                          rank:   0    1    2    3    4    5    6    7
    // CHECK: util.switch index from [%c0, %c1, %c0, %c1, %c0, %c1, %c0, %c1] at %channel_rank else %c-1 : index
    // CHECK: util.switch index from [%c0, %c0, %c1, %c1, %c2, %c2, %c3, %c3] at %channel_rank else %c-1 : index
    %out = "mhlo.all_reduce"(%input) ({
      ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
        %sum = mhlo.add %arg0, %arg1 : tensor<f32>
        mhlo.return %sum : tensor<f32>
      }) {channel_handle = #mhlo.channel_handle<handle = 0, type = 1>,
          replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
         } : (tensor<2304xf32>) -> tensor<2304xf32>
    return %out : tensor<2304xf32>
  }
}

// -----

// cross_replica_and_partition: channel_id > 0 && use_global_device_ids = false
module @jit_fn attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 4 : i32 } {
  // CHECK-LABEL: @cross_replica_and_partition
  func.func @cross_replica_and_partition(%input : tensor<2304xf32>) -> tensor<2304xf32> {
    // Cross replica_and_partition should form groups (0,2,1,3),(4,6,5,7), where each number represents a cell below.
    // Note that the rank is assigned in a partiton first, e.g., rank 0 and 1 are assigned to cell 0 and 2, respectively.
    // +---+---+
    // | 0   1 |
    // | 2   3 |
    // |---+---|
    // | 4   5 |
    // | 6   7 |
    // +---+---+
    //                          rank:   0    1    2    3    4    5    6    7
    // CHECK: util.switch index from [%c0, %c0, %c0, %c0, %c1, %c1, %c1, %c1] at %channel_rank else %c-1 : index
    // CHECK: util.switch index from [%c0, %c2, %c1, %c3, %c0, %c2, %c1, %c3] at %channel_rank else %c-1 : index
    %out = "mhlo.all_reduce"(%input) ({
      ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
        %sum = mhlo.add %arg0, %arg1 : tensor<f32>
        mhlo.return %sum : tensor<f32>
      }) {channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
          replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>
         } : (tensor<2304xf32>) -> tensor<2304xf32>
    return %out : tensor<2304xf32>
  }
}

// -----

// cross_partition: channel_id > 0
module @jit_fn attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 4 : i32 } {
  // CHECK-LABEL: @cross_partition
  func.func @cross_partition(%input : tensor<2304xf32>) -> tensor<2304xf32> {
    // Cross partition should form groups (0,1),(2,3),(4,5),(6,7) where each number represents a cell below.
    // +---+---+
    // | 0   1 |
    // +---+---+
    // | 2   3 |
    // +---+---+
    // | 4   5 |
    // +---+---+
    // | 6   7 |
    // +---+---+
    //                          rank:   0    1    2    3    4    5    6    7
    // CHECK: util.switch index from [%c0, %c0, %c1, %c1, %c2, %c2, %c3, %c3] at %channel_rank else %c-1 : index
    // CHECK: util.switch index from [%c0, %c1, %c0, %c1, %c0, %c1, %c0, %c1] at %channel_rank else %c-1 : index
    %out = "mhlo.all_to_all"(%input) {
      split_dimension = 0 : i64,
      concat_dimension = 0 : i64,
      split_count = 2 : i64,
      channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<2304xf32>) -> tensor<2304xf32>
    return %out : tensor<2304xf32>
  }
}

// -----

// CHECK-LABEL: @collective_permute
// CHECK-SAME: (%[[ARG0:.+]]: tensor<8xf32>) -> tensor<8xf32>
func.func @collective_permute(%input : tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: %[[CHANNEL:.+]] = flow.channel.default : !flow.channel
  // CHECK: %[[RANK:.+]] = flow.channel.rank %[[CHANNEL]] : index
  // CHECK: %[[SEND:.+]] = util.switch index from [%c1, %c2, %c3, %c0] at %[[RANK]] else %c-1
  // CHECK: %[[RECV:.+]] = util.switch index from [%c3, %c0, %c1, %c2] at %[[RANK]] else %c-1
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<8xf32>
  // CHECK: %[[OP:.+]] = flow.collective.send_recv f32, %[[EMPTY]], %[[ARG0]], %[[CHANNEL]], %[[SEND]], %[[RECV]] : (tensor<8xf32>, tensor<8xf32>, !flow.channel, index, index) -> %[[EMPTY]] as tensor<8xf32>
  // CHECK: return %[[OP]] : tensor<8xf32>
  %out = "mhlo.collective_permute"(%input) {
        source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [3, 0]]> : tensor<4x2xi64>,
        channel_handle = #mhlo.channel_handle<handle = 1, type = 1>} : (tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

// collective_permute cross_replica: channel_id <= 0
module @jit_fn attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 4 : i32 } {
  // CHECK-LABEL: @collective_permute_cross_replica
  func.func @collective_permute_cross_replica(%input : tensor<8xf32>) -> tensor<8xf32> {
    // Cross replica should form groups (0,2,4,6),(1,3,5,7) where each number represents a cell below.
    // +---+---+
    // | 0 | 1 |
    // |   |   |
    // | 2 | 3 |
    // |   |   |
    // | 4 | 5 |
    // |   |   |
    // | 6 | 7 |
    // +---+---+
    //                          rank:   0    1    2    3    4    5    6    7
    // CHECK: util.switch index from [%c0, %c1, %c0, %c1, %c0, %c1, %c0, %c1] at %channel_rank else %c-1 : index
    // CHECK: util.switch index from [%c0, %c0, %c1, %c1, %c2, %c2, %c3, %c3] at %channel_rank else %c-1 : index
    %out = "mhlo.collective_permute"(%input) {
          source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [3, 0]]> : tensor<4x2xi64>,
          channel_handle = #mhlo.channel_handle<handle = 0, type = 1>} : (tensor<8xf32>) -> tensor<8xf32>
    return %out : tensor<8xf32>
  }
}

// -----

// collective_permute cross_partition: channel_id > 0
module @jit_fn attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 4 : i32 } {
  // CHECK-LABEL: @collective_permute_cross_partition
  func.func @collective_permute_cross_partition(%input : tensor<8xf32>) -> tensor<8xf32> {
    // Cross partition should form groups (0,1),(2,3),(4,5),(6,7) where each number represents a cell below.
    // +---+---+
    // | 0   1 |
    // +---+---+
    // | 2   3 |
    // |---+---|
    // | 4   5 |
    // +---+---+
    // | 6   7 |
    // +---+---+
    //                          rank:   0    1    2    3    4    5    6    7
    // CHECK: util.switch index from [%c0, %c0, %c1, %c1, %c2, %c2, %c3, %c3] at %channel_rank else %c-1 : index
    // CHECK: util.switch index from [%c0, %c1, %c0, %c1, %c0, %c1, %c0, %c1] at %channel_rank else %c-1 : index
    %out = "mhlo.collective_permute"(%input) {
          source_target_pairs = dense<[[0, 1]]> : tensor<1x2xi64>,
          channel_handle = #mhlo.channel_handle<handle = 1, type = 1>} : (tensor<8xf32>) -> tensor<8xf32>
    return %out : tensor<8xf32>
  }
}
