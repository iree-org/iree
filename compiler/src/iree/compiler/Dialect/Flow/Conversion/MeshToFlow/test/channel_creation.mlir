// RUN: iree-opt --split-input-file --iree-convert-mesh-to-flow --cse %s | FileCheck %s

// CHECK-LABEL: module @static_1d_mesh_grouping_along_axis_0
module @static_1d_mesh_grouping_along_axis_0 {

  // No channel initialization default channel is expected.
  // CHECK-NOT: util.global private @_mesh_mesh_1d_axes_0 {inlining_policy = #util.inline.never} : !flow.channel
  mesh.cluster @mesh_1d(shape = 2)

  func.func @f(
      %arg0 : tensor<1xi8>) -> tensor<1xi8> {
    %0 = mesh.all_reduce %arg0 on @mesh_1d mesh_axes = [0] reduction = <sum>
      : tensor<1xi8> -> tensor<1xi8>
    return %0 : tensor<1xi8>
  }
}

// -----

// CHECK-LABEL: module @static_2d_mesh_grouping_along_axis_1
module @static_2d_mesh_grouping_along_axis_1 {

  // CHECK: util.global private @_mesh_mesh_2d_axes_1 {inlining_policy = #util.inline.never} : !flow.channel
  // CHECK: util.initializer {
    // CHECK-DAG: %[[AXIS_1_SIZE:.*]] = arith.constant 4 : index
    // CHECK-DAG: %[[AXIS_0_SIZE:.*]] = arith.constant 3 : index
    // CHECK-DAG: %[[DEFAULT_CHANNEL:.*]] = flow.channel.default : !flow.channel
    // CHECK: %[[CHANNEL_RANK:.*]] = flow.channel.rank %[[DEFAULT_CHANNEL]] : index
    // CHECK: %[[COLOR_AND_KEY:.*]]:2 = affine.delinearize_index %[[CHANNEL_RANK]] into
    // CHECK-SAME: (%[[AXIS_0_SIZE]], %[[AXIS_1_SIZE]]) : index, index
    // CHECK: %[[CHANNEL:.*]] = flow.channel.split
    // CHECK-SAME: %[[DEFAULT_CHANNEL]], %[[COLOR_AND_KEY]]#0, %[[COLOR_AND_KEY]]#1 : !flow.channel -> !flow.channel
    // CHECK: util.global.store %[[CHANNEL]], @_mesh_mesh_2d_axes_1 : !flow.channel
  mesh.cluster @mesh_2d(shape = 3x4)

  func.func @f(%input : tensor<1xi8>) -> tensor<1xi8> {
    %out = mesh.all_reduce %input on @mesh_2d mesh_axes = [1] : tensor<1xi8> -> tensor<1xi8>
    return %out : tensor<1xi8>
  }
}

// -----

// CHECK: #map = affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>

// CHECK-LABEL: module @static_4d_mesh_grouping_along_axes_2_1
module @static_4d_mesh_grouping_along_axes_2_1 {

  // CHECK: util.global private @_mesh_mesh_4d_axes_2_1 {inlining_policy = #util.inline.never} : !flow.channel
  // CHECK: util.initializer {
    // CHECK-DAG: %[[AXIS_3_SIZE:.*]] = arith.constant 6 : index
    // CHECK-DAG: %[[AXIS_2_SIZE:.*]] = arith.constant 5 : index
    // CHECK-DAG: %[[AXIS_1_SIZE:.*]] = arith.constant 4 : index
    // CHECK-DAG: %[[AXIS_0_SIZE:.*]] = arith.constant 3 : index
    // CHECK-DAG: %[[DEFAULT_CHANNEL:.*]] = flow.channel.default : !flow.channel
    // CHECK: %[[CHANNEL_RANK:.*]] = flow.channel.rank %[[DEFAULT_CHANNEL]] : index
    // CHECK: %[[DEVICE_MULTI_IDX:.*]]:4 = affine.delinearize_index %[[CHANNEL_RANK]] into
    // CHECK-SAME: (%[[AXIS_0_SIZE]], %[[AXIS_1_SIZE]], %[[AXIS_2_SIZE]], %[[AXIS_3_SIZE]]) : index, index, index, index
    // CHECK: %[[IN_GROUP_IDX:.*]] = affine.apply
    // CHECK-SAME: #map()[%[[DEVICE_MULTI_IDX]]#2, %[[AXIS_1_SIZE]], %[[DEVICE_MULTI_IDX]]#1]
    // CHECK: %[[GROUP_IDX:.*]] = affine.apply
    // CHECK-SAME: #map()[%[[DEVICE_MULTI_IDX]]#0, %[[AXIS_3_SIZE]], %[[DEVICE_MULTI_IDX]]#3]
    // CHECK: %[[CHANNEL:.*]] = flow.channel.split
    // CHECK-SAME: %[[DEFAULT_CHANNEL]], %[[GROUP_IDX]], %[[IN_GROUP_IDX]] : !flow.channel -> !flow.channel
    // CHECK: util.global.store %[[CHANNEL]], @_mesh_mesh_4d_axes_2_1 : !flow.channel
  mesh.cluster @mesh_4d(shape = 3x4x5x6)

  func.func @f(%input : tensor<1xi8>) -> tensor<1xi8> {
    %out = mesh.all_reduce %input on @mesh_4d mesh_axes = [2, 1] : tensor<1xi8> -> tensor<1xi8>
    return %out : tensor<1xi8>
  }
}

// -----

// CHECK-LABEL: module @multiple_different_channels
module @multiple_different_channels {

  // CHECK-DAG: util.global private @_mesh_mesh_2d_axes_0 {inlining_policy = #util.inline.never} : !flow.channel
  // CHECK-DAG: util.global private @_mesh_mesh_2d_axes_1 {inlining_policy = #util.inline.never} : !flow.channel
  mesh.cluster @mesh_2d(shape = 3x4)

  func.func @f(%input : tensor<1xi8>) -> (tensor<1xi8>, tensor<1xi8>) {
    %out0 = mesh.all_reduce %input on @mesh_2d mesh_axes = [0] : tensor<1xi8> -> tensor<1xi8>
    %out1 = mesh.all_reduce %input on @mesh_2d mesh_axes = [1] : tensor<1xi8> -> tensor<1xi8>
    return %out0, %out1 : tensor<1xi8>, tensor<1xi8>
  }
}

// -----

// CHECK-LABEL: module @same_channel_used_multiple_times
module @same_channel_used_multiple_times {

  // CHECK: util.global private @_mesh_mesh_2d_axes_0 {inlining_policy = #util.inline.never} : !flow.channel
  mesh.cluster @mesh_2d(shape = 3x4)

  func.func @f(%input0 : tensor<1xi8>, %input1 : tensor<1xi8>) -> (tensor<1xi8>, tensor<1xi8>) {
    %out0 = mesh.all_reduce %input0 on @mesh_2d mesh_axes = [0] : tensor<1xi8> -> tensor<1xi8>
    %out1 = mesh.all_reduce %input1 on @mesh_2d mesh_axes = [0] : tensor<1xi8> -> tensor<1xi8>
    return %out0, %out1 : tensor<1xi8>, tensor<1xi8>
  }
}

// -----

// CHECK-LABEL: module @multiple_meshes
module @multiple_meshes {

  // CHECK: util.global private @_mesh_mesh1_axes_0 {inlining_policy = #util.inline.never} : !flow.channel
  // CHECK: util.initializer {
    // CHECK: %[[DEFAULT_CHANNEL:.*]] = flow.channel.default "mesh1" : !flow.channel
    // CHECK: %[[CHANNEL:.*]] = flow.channel.split
    // CHECK-SAME: %[[DEFAULT_CHANNEL]], %{{.*}}, %{{.*}} : !flow.channel -> !flow.channel
    // CHECK: util.global.store %[[CHANNEL]], @_mesh_mesh1_axes_0 : !flow.channel
  mesh.cluster @mesh1(shape = 1x2)
  // CHECK: util.global private @_mesh_mesh2_axes_1 {inlining_policy = #util.inline.never} : !flow.channel
  // CHECK: util.initializer {
    // CHECK: %[[DEFAULT_CHANNEL:.*]] = flow.channel.default "mesh2" : !flow.channel
    // CHECK: %[[CHANNEL:.*]] = flow.channel.split
    // CHECK-SAME: %[[DEFAULT_CHANNEL]], %{{.*}}, %{{.*}} : !flow.channel -> !flow.channel
    // CHECK: util.global.store %[[CHANNEL]], @_mesh_mesh2_axes_1 : !flow.channel
  mesh.cluster @mesh2(shape = 3x4)

  func.func @f(%input0 : tensor<1xi8>, %input1 : tensor<1xi8>) -> (tensor<1xi8>, tensor<1xi8>) {
    %out0 = mesh.all_reduce %input0 on @mesh1 mesh_axes = [0] : tensor<1xi8> -> tensor<1xi8>
    %out1 = mesh.all_reduce %input1 on @mesh2 mesh_axes = [1] : tensor<1xi8> -> tensor<1xi8>
    return %out0, %out1 : tensor<1xi8>, tensor<1xi8>
  }
}
