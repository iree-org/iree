// RUN: iree-opt -split-input-file -iree-codegen-linalg-tile-and-fuse %s | IreeFileCheck %s

// Test to check that convolution with padding is not tiled.
module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
    [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @conv_padding(%arg0 : memref<?x?x?x?xf32>, %arg1 : memref<?x?x?x?xf32>,
                     %arg2 : memref<?x?x?x?xf32>) {
    linalg.conv(%arg0, %arg1, %arg2)
      {dilations = [1, 1],
       padding = dense<[[1, 1], [0, 1]]> : tensor<2x2xi64>, strides = [1, 1]} :
      memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
    return
  }
}
// CHECK-LABEL: func @conv_padding
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//       CHECK:   linalg.conv
//  CHECK-SAME:     %[[ARG0]]
//  CHECK-SAME:     %[[ARG1]]
//  CHECK-SAME:     %[[ARG2]]

// -----

module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
    [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @conv_no_padding(%arg0 : memref<?x?x?x?xf32>, %arg1 : memref<?x?x?x?xf32>,
                        %arg2 : memref<?x?x?x?xf32>) {
    linalg.conv(%arg0, %arg1, %arg2) {dilations = [1, 1], strides = [1, 1]} :
      memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
    return
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 4)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 32)>
//       CHECK: func @conv_no_padding
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   local_size = dense<[32, 4, 1]>
//  CHECK-SAME:   vkspv.workgroup_count_from_result_shape = 0
//   CHECK-DAG:   %[[C0:.+]] = constant 0
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[NBLOCKSX:.+]] = "gpu.grid_dim"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-DAG:   %[[NBLOCKSY:.+]] = "gpu.grid_dim"() {dimension = "y"}
//   CHECK-DAG:   %[[BIDZ:.+]] = "gpu.block_id"() {dimension = "z"}
//   CHECK-DAG:   %[[NBLOCKSZ:.+]] = "gpu.grid_dim"() {dimension = "z"}
//       CHECK:   %[[LBY:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[STEPY:.+]] = affine.apply #[[MAP0]]()[%[[NBLOCKSY]]]
//       CHECK:   %[[LBX:.+]] = affine.apply #[[MAP1]]()[%[[BIDX]]
//       CHECK:   %[[STEPX:.+]] = affine.apply #[[MAP1]]()[%[[NBLOCKSX]]]
//       CHECK:   scf.parallel (%[[IV0:.+]], %[[IV1:.+]], %[[IV2:.+]]) =
//       (%[[BIDZ]], %[[LBY]], %[[LBX]])
//  CHECK-SAME:      step (%[[NBLOCKSZ]], %[[STEPY]], %[[STEPX]])
//       CHECK:     %[[VIEW1:.+]] = subview %[[ARG1]][%[[IV0]], %[[IV1]],
//       %[[IV2]], %[[C0]]] CHECK:     %[[VIEW2:.+]] = subview
//       %[[ARG2]][%[[IV0]], %[[IV1]], %[[IV2]], %[[C0]]] CHECK:     linalg.conv
//  CHECK-SAME:       %[[ARG0]], %[[VIEW1]], %[[VIEW2]]
//  CHECK-SAME:       "workgroup"

// -----

module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
    [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @matmul(%arg0: memref<?x?xf32>,
                %arg1: memref<?x?xf32>,
                %ret0: memref<?x?xf32>) {
    linalg.matmul %arg0, %arg1, %ret0 :
      (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>)
    return
  }
}

//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 8)>
//       CHECK: func @matmul
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//  CHECK-SAME:   local_size = dense<[8, 8, 1]>
//  CHECK-SAME:   vkspv.workgroup_count_from_result_shape = 2
//   CHECK-DAG:   %[[C0:.+]] = constant 0
//   CHECK-DAG:   %[[C4:.+]] = constant 4
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-NOT:   scf.parallel
//       CHECK:   scf.for %[[IV:.+]] = %[[C0]] to %{{.+}} step %[[C4]]
//       CHECK:     %[[LBY:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:     %[[VIEW0:.+]] = subview %[[ARG0]][%[[LBY]], %[[IV]]
//       CHECK:     %[[LBX:.+]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//       CHECK:     %[[VIEW1:.+]] = subview %[[ARG1]][%[[IV]], %[[LBX]]]
//       CHECK:     %[[LBY_2:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:     %[[LBX_2:.+]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//       CHECK:     %[[VIEW2:.+]] = subview %[[ARG2]][%[[LBY_2]], %[[LBX_2]]]
//       CHECK:     linalg.matmul
//  CHECK-SAME:       "workgroup_numprocs_ge_numiters"
//  CHECK-SAME:       %[[VIEW0]], %[[VIEW1]], %[[VIEW2]]

// -----

module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
    [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @pooling_sum_no_padding(%arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>,
                               %arg2 : memref<?x?xf32>) {
    linalg.pooling_max(%arg0, %arg1, %arg2) {dilations = [1, 1], strides = [1, 1]} :
      memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
    return
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 4)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 32)>
//       CHECK: func @pooling_sum_no_padding
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//  CHECK-SAME:   local_size = dense<[32, 4, 1]>
//  CHECK-SAME:   vkspv.workgroup_count_from_result_shape = 0
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[NBLOCKSX:.+]] = "gpu.grid_dim"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-DAG:   %[[NBLOCKSY:.+]] = "gpu.grid_dim"() {dimension = "y"}
//       CHECK:   %[[LBY:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[STEPY:.+]] = affine.apply #[[MAP0]]()[%[[NBLOCKSY]]]
//       CHECK:   %[[LBX:.+]] = affine.apply #[[MAP1]]()[%[[BIDX]]
//       CHECK:   %[[STEPX:.+]] = affine.apply #[[MAP1]]()[%[[NBLOCKSX]]]
//       CHECK:   scf.parallel (%[[IV0:.+]], %[[IV1:.+]]) = (%[[LBY]], %[[LBX]])
//  CHECK-SAME:      step (%[[STEPY]], %[[STEPX]])
//       CHECK:     %[[VIEW0:.+]] = subview %[[ARG0]][%[[IV0]], %[[IV1]]]
//       CHECK:     %[[VIEW2:.+]] = subview %[[ARG2]][%[[IV0]], %[[IV1]]]
//       CHECK:     linalg.pooling_max
//  CHECK-SAME:       %[[VIEW0]], %[[ARG1]], %[[VIEW2]]
//  CHECK-SAME:       "workgroup"
