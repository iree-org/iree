// RUN: iree-opt --split-input-file --iree-convert-mesh-to-flow --cse %s | FileCheck %s

mesh.mesh @mesh_2d(shape = 3x4)

// CHECK-LABEL: util.func public @all_gather_non_default_channel
 util.func public @all_gather_non_default_channel(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x4xi8>
    %arg0 : tensor<3x4xi8>) -> tensor<3x16xi8> {
  // CHECK-DAG: %[[CHANNEL:.*]] = util.global.load @_mesh_mesh_2d_axes_1 : !flow.channel
  // CHECK-DAG: %[[TRANSPOSED_OPERAND_INIT_VAL:.*]] = tensor.empty() : tensor<4x3xi8>
  // CHECK: %[[TRANSPOSED_OPERAND:.*]] = linalg.transpose
  // CHECK-SAME: ins(%[[ARG]] : tensor<3x4xi8>) outs(%[[TRANSPOSED_OPERAND_INIT_VAL]] : tensor<4x3xi8>) permutation = [1, 0]
  // CHECK: %[[ALL_GATHER_INITIAL_VAL:.*]] = tensor.empty() : tensor<16x3xi8>
  // CHECK: %[[ALL_GATHER_RES:.*]] = flow.collective.all_gather ui8,
  // CHECK-SAME: %[[ALL_GATHER_INITIAL_VAL]], %[[TRANSPOSED_OPERAND]], %[[CHANNEL]]
  // CHECK-SAME: (tensor<16x3xi8>, tensor<4x3xi8>, !flow.channel) -> %[[ALL_GATHER_INITIAL_VAL]] as tensor<16x3xi8>
  // CHECK: %[[RES_INIT_VAL:.*]] = tensor.empty() : tensor<3x16xi8>
  // CHECK: %[[RES:.*]] = linalg.transpose
  // CHECK-SAME: ins(%[[ALL_GATHER_RES]] : tensor<16x3xi8>) outs(%[[RES_INIT_VAL]] : tensor<3x16xi8>) permutation = [1, 0]
  %0 = mesh.all_gather %arg0 on @mesh_2d mesh_axes = [1] gather_axis = 1
    : tensor<3x4xi8> -> tensor<3x16xi8>
  // CHECK: util.return %[[RES]] : tensor<3x16xi8>
  util.return %0 : tensor<3x16xi8>
}

// -----

mesh.mesh @mesh_1d(shape = 2)

// CHECK-LABEL: util.func public @all_reduce_sum_default_channel
 util.func public @all_reduce_sum_default_channel(
    // CHECK-SAME: %[[ARG:.*]]: tensor<1xi8>
    %arg0 : tensor<1xi8>) -> tensor<1xi8> {
  // CHECK: %[[CHANNEL:.*]] = flow.channel.default : !flow.channel
  // CHECK: %[[INITIAL_VAL:.*]] = tensor.empty() : tensor<1xi8>
  // CHECK: %[[RES:.*]] = flow.collective.all_reduce sum, ui8, %[[INITIAL_VAL]], %[[ARG]], %[[CHANNEL]]
  // CHECK-SAME: (tensor<1xi8>, tensor<1xi8>, !flow.channel) -> %[[INITIAL_VAL]] as tensor<1xi8>
  %0 = mesh.all_reduce %arg0 on @mesh_1d mesh_axes = [0]
    : tensor<1xi8> -> tensor<1xi8>
  // CHECK: util.return %[[RES]] : tensor<1xi8>
  util.return %0 : tensor<1xi8>
}

// -----

mesh.mesh @mesh_2d(shape = 2x2)

// CHECK-LABEL: util.func public @all_reduce_min_non_default_channel
 util.func public @all_reduce_min_non_default_channel(
    // CHECK-SAME: %[[ARG:.*]]: tensor<1xi8>
    %arg0 : tensor<1xi8>) -> tensor<1xi8> {
  // CHECK-DAG: %[[CHANNEL:.*]] = util.global.load @_mesh_mesh_2d_axes_1_0 : !flow.channel
  // CHECK-DAG: %[[INITIAL_VAL:.*]] = tensor.empty() : tensor<1xi8>
  // CHECK: %[[RES:.*]] = flow.collective.all_reduce minimum, ui8, %[[INITIAL_VAL]], %[[ARG]], %[[CHANNEL]]
  // CHECK-SAME: (tensor<1xi8>, tensor<1xi8>, !flow.channel) -> %[[INITIAL_VAL]] as tensor<1xi8>
  %0 = mesh.all_reduce %arg0 on @mesh_2d mesh_axes = [1, 0] reduction = <min>
    : tensor<1xi8> -> tensor<1xi8>
  // CHECK: util.return %[[RES]] : tensor<1xi8>
  util.return %0 : tensor<1xi8>
}

// -----

mesh.mesh @mesh_1d(shape = 2)

// CHECK-LABEL: util.func public @all_reduce_f32
 util.func public @all_reduce_f32(
    // CHECK-SAME: %[[ARG:.*]]: tensor<1xf32>
    %arg0 : tensor<1xf32>) -> tensor<1xf32> {
  // CHECK-DAG: %[[CHANNEL:.*]] = flow.channel.default : !flow.channel
  // CHECK-DAG: %[[INITIAL_VAL:.*]] = tensor.empty() : tensor<1xf32>
  // CHECK: %[[RES:.*]] = flow.collective.all_reduce sum, f32, %[[INITIAL_VAL]], %[[ARG]], %[[CHANNEL]]
  // CHECK-SAME: (tensor<1xf32>, tensor<1xf32>, !flow.channel) -> %[[INITIAL_VAL]] as tensor<1xf32>
  %0 = mesh.all_reduce %arg0 on @mesh_1d mesh_axes = [0]
    : tensor<1xf32> -> tensor<1xf32>
  // CHECK: util.return %[[RES]] : tensor<1xf32>
  util.return %0 : tensor<1xf32>
}

// -----

mesh.mesh @mesh_1d(shape = 2)

// CHECK-LABEL: util.func public @process_linear_index
 util.func public @process_linear_index() -> index {
  // CHECK: %[[CHANNEL:.*]] = flow.channel.default : !flow.channel
  // CHECK: %[[RES:.*]] = flow.channel.rank %[[CHANNEL]] : index
  %0 = mesh.process_linear_index on @mesh_1d : index
  // CHECK: util.return %[[RES]] : index
  util.return %0 : index
}

// -----

mesh.mesh @mesh_3d(shape = 2x3x4)

// CHECK-LABEL: util.func public @all_to_all_non_default_channel
 util.func public @all_to_all_non_default_channel(
    // CHECK-SAME: %[[ARG:.*]]: tensor<1x12x3x4x5xf32>
    %arg0 : tensor<1x12x3x4x5xf32>) -> tensor<1x2x3x24x5xf32> {
  // CHECK: %[[CHANNEL:.*]] = util.global.load @_mesh_mesh_3d_axes_1_0 : !flow.channel
  // CHECK: %[[SPLIT_AXIS_AT_FRONT:.*]] = linalg.transpose ins(%[[ARG]] : tensor<1x12x3x4x5xf32>)
  // CHECK-SAME: outs(%{{.*}} : tensor<12x1x3x4x5xf32>) permutation = [1, 0, 2, 3, 4]
  // CHECK: %[[FLOW_ALL_TO_ALL:.*]] = flow.collective.all_to_all f32, %{{.*}}, %[[SPLIT_AXIS_AT_FRONT]], %_mesh_mesh_3d_axes_1_0 :
  // CHECK-SAME: (tensor<12x1x3x4x5xf32>, tensor<12x1x3x4x5xf32>, !flow.channel) -> %0 as tensor<12x1x3x4x5xf32>
  // CHECK: %[[SPLIT_AXIS_BACK_IN_ITS_PLACE:.*]] = linalg.transpose ins(%1 : tensor<12x1x3x4x5xf32>)
  // CHECK-SAME: outs(%{{.*}} : tensor<1x12x3x4x5xf32>) permutation = [1, 0, 2, 3, 4]
  // CHECK: %[[SPLIT_AXIS_IS_SPLIT:.*]] = tensor.expand_shape %[[SPLIT_AXIS_BACK_IN_ITS_PLACE]]
  // CHECK-SAME-LITERAL: [[0], [1, 2], [3], [4], [5]] : tensor<1x12x3x4x5xf32> into tensor<1x6x2x3x4x5xf32>
  // CHECK: %[[MOVED_SPLIT_COUNT_AXIS:.*]] = linalg.transpose ins(%[[SPLIT_AXIS_IS_SPLIT]] : tensor<1x6x2x3x4x5xf32>)
  // CHECK-SAME: outs(%{{.*}} : tensor<1x2x3x6x4x5xf32>) permutation = [0, 2, 3, 1, 4, 5]
  // CHECK: %[[COLLAPSED_SPLIT_COUNT_INTO_CONCAT_AXIS:.*]] = tensor.collapse_shape %[[MOVED_SPLIT_COUNT_AXIS]]
  // CHECK-SAME-LITERAL: [[0], [1], [2], [3, 4], [5]] : tensor<1x2x3x6x4x5xf32> into tensor<1x2x3x24x5xf32>
  %0 = mesh.all_to_all %arg0 on @mesh_3d mesh_axes = [1, 0] split_axis = 1 concat_axis = 3
    : tensor<1x12x3x4x5xf32> -> tensor<1x2x3x24x5xf32>
  // CHECK: util.return %[[COLLAPSED_SPLIT_COUNT_INTO_CONCAT_AXIS]] : tensor<1x2x3x24x5xf32>
  util.return %0 : tensor<1x2x3x24x5xf32>
}

// -----

mesh.mesh @mesh_2d(shape = 2x2)

// CHECK-LABEL: util.func public @reduce_scatter_non_default_channel
 util.func public @reduce_scatter_non_default_channel(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x2xi8>
    %arg0 : tensor<3x2xi8>) -> tensor<3x1xi8> {
  // CHECK-DAG: %[[CHANNEL:.*]] = util.global.load @_mesh_mesh_2d_axes_0 : !flow.channel
  // CHECK-DAG: %[[TRANSPOSED_OPERAND_INIT_VAL:.*]] = tensor.empty() : tensor<2x3xi8>
  // CHECK: %[[TRANSPOSED_OPERAND:.*]] = linalg.transpose
  // CHECK-SAME: ins(%[[ARG]] : tensor<3x2xi8>) outs(%[[TRANSPOSED_OPERAND_INIT_VAL]] : tensor<2x3xi8>) permutation = [1, 0]
  // CHECK: %[[REDUCE_SCATTER_INITIAL_VAL:.*]] = tensor.empty() : tensor<1x3xi8>
  // CHECK: %[[REDUCE_SCATTER_RES:.*]] = flow.collective.reduce_scatter sum, ui8,
  // CHECK-SAME: %[[REDUCE_SCATTER_INITIAL_VAL]], %[[TRANSPOSED_OPERAND]], %[[CHANNEL]]
  // CHECK-SAME: (tensor<1x3xi8>, tensor<2x3xi8>, !flow.channel) -> %[[REDUCE_SCATTER_INITIAL_VAL]] as tensor<1x3xi8>
  // CHECK: %[[RES_INIT_VAL:.*]] = tensor.empty() : tensor<3x1xi8>
  // CHECK: %[[RES:.*]] = linalg.transpose
  // CHECK-SAME: ins(%[[REDUCE_SCATTER_RES]] : tensor<1x3xi8>) outs(%[[RES_INIT_VAL]] : tensor<3x1xi8>) permutation = [1, 0]
  %0 = mesh.reduce_scatter %arg0 on @mesh_2d mesh_axes = [0] scatter_axis = 1
    : tensor<3x2xi8> -> tensor<3x1xi8>
  // CHECK: util.return %[[RES]] : tensor<3x1xi8>
  util.return %0 : tensor<3x1xi8>
}
