// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-codegen-gpu-fuse-subgroup-consumers)" --split-input-file | FileCheck %s

// Test that the pass fuses a consumer into a subgroup-scoped pcf.generic.
func.func @fuse_consumer_into_subgroup_generic(
    %arg0: tensor<?x?xi32>, %dest: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = pcf.generic scope(#iree_gpu.subgroup_scope)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, sync(#iree_gpu.subgroup_scope)>)
        -> (tensor<?x?xi32>) {
    %cst = arith.constant dense<5> : tensor<4x5xi32>
    pcf.write_slice %cst into %ref[%id, 0] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<?x?xi32, sync(#iree_gpu.subgroup_scope)>
    pcf.return
  }
  %1 = linalg.copy ins(%0: tensor<?x?xi32>) outs(%dest: tensor<?x?xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: @fuse_consumer_into_subgroup_generic
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<?x?xi32>
//       CHECK:   pcf.generic scope(#iree_gpu.subgroup_scope)
//       CHECK:     linalg.copy
//       CHECK:     pcf.write_slice
//       CHECK:     pcf.return

// -----

// Test that non-subgroup-scoped pcf.generic ops are NOT fused.
func.func @no_fuse_lane_scoped_generic(
    %arg0: tensor<?x?xi32>, %dest: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = pcf.generic scope(#iree_gpu.lane_scope)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, sync(#iree_gpu.lane_scope)>)
        -> (tensor<?x?xi32>) {
    %cst = arith.constant dense<5> : tensor<4x5xi32>
    pcf.write_slice %cst into %ref[%id, 0] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<?x?xi32, sync(#iree_gpu.lane_scope)>
    pcf.return
  }
  %1 = linalg.copy ins(%0: tensor<?x?xi32>) outs(%dest: tensor<?x?xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: @no_fuse_lane_scoped_generic
//       CHECK:   %[[G:.+]] = pcf.generic scope(#iree_gpu.lane_scope)
//   CHECK-NOT:     linalg.copy
//       CHECK:     pcf.return
//       CHECK:   linalg.copy ins(%[[G]]

// -----

// Test that extract_slice on a subgroup-scoped pcf.generic result is fused.
func.func @fuse_extract_slice_into_subgroup_generic(
    %arg0: tensor<16x32xi32>) -> tensor<16x16xi32> {
  %0 = pcf.generic scope(#iree_gpu.subgroup_scope)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<16x32xi32, sync(#iree_gpu.subgroup_scope)>)
        -> (tensor<16x32xi32>) {
    %cst = arith.constant dense<5> : tensor<4x8xi32>
    pcf.write_slice %cst into %ref[%id, 0] [4, 8] [1, 1] : tensor<4x8xi32> into !pcf.sref<16x32xi32, sync(#iree_gpu.subgroup_scope)>
    pcf.return
  }
  %1 = tensor.extract_slice %0[0, 0] [16, 16] [1, 1] : tensor<16x32xi32> to tensor<16x16xi32>
  return %1 : tensor<16x16xi32>
}

// CHECK-LABEL: @fuse_extract_slice_into_subgroup_generic
//       CHECK:   pcf.generic scope(#iree_gpu.subgroup_scope)
//       CHECK:     -> (tensor<16x16xi32>)
//       CHECK:     pcf.write_slice
//       CHECK:     pcf.return
//   CHECK-NOT:   tensor.extract_slice
