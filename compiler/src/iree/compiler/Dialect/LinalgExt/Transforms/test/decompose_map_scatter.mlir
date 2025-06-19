// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-map-scatter,cse))" --split-input-file %s | FileCheck %s

func.func @identity_map_scatter(
    %input: vector<4x16xf32>, %output: memref<4x16xf32>
) {
  iree_linalg_ext.map_scatter %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : vector<4x16xf32> into memref<4x16xf32>
  return
}
// CHECK-LABEL: func.func @identity_map_scatter(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[FLAT_OUTPUT:.+]] = memref.collapse_shape %[[OUTPUT]] {{.*}} memref<4x16xf32> into memref<64xf32>
//   CHECK-DAG:   %[[FLAT_INDICES:.+]] = vector.shape_cast{{.*}} : vector<4x16xindex> to vector<64xindex>
//   CHECK-DAG:   %[[FLAT_MASK:.+]] = vector.shape_cast{{.*}} : vector<4x16xi1> to vector<64xi1>
//   CHECK-DAG:   %[[FLAT_INPUT:.+]] = vector.shape_cast %[[INPUT]] : vector<4x16xf32> to vector<64xf32>
//       CHECK:   vector.scatter %[[FLAT_OUTPUT]][%[[C0]]]
//  CHECK-SAME:     [%[[FLAT_INDICES]]], %[[FLAT_MASK]], %[[FLAT_INPUT]]

// -----

func.func @map_scatter_with_linearize_delinearize_idx(
    %input: vector<2x2x64xf32>, %output: memref<4x32x2xf32>
) {
  iree_linalg_ext.map_scatter %input into %output {
    ^bb0(%idx0: index, %idx1: index, %idx2: index):
      %mask = arith.constant true
      %out_idx_0 = affine.linearize_index [%idx0, %idx1] by (2, 2) : index
      %out_idx:2 = affine.delinearize_index %idx2 into (32, 2) : index, index
      iree_linalg_ext.yield %out_idx_0, %out_idx#0, %out_idx#1, %mask : index, index, index, i1
  } : vector<2x2x64xf32> into memref<4x32x2xf32>
  return
}
// CHECK-LABEL: func.func @map_scatter_with_linearize_delinearize_idx(
//   CHECK-NOT:   iree_linalg_ext.map_scatter
//       CHECK:   vector.scatter

// -----

func.func @map_scatter_with_mask(
    %input: vector<64xf32>, %output: memref<?xf32>
) {
  %c0 = arith.constant 0 : index
  %dim = memref.dim %output, %c0 : memref<?xf32>
  iree_linalg_ext.map_scatter %input into %output {
    ^bb0(%idx0: index):
      %mask = arith.cmpi uge, %idx0, %dim : index
      iree_linalg_ext.yield %idx0, %mask : index, i1
  } : vector<64xf32> into memref<?xf32>
  return
}
// CHECK-LABEL: func.func @map_scatter_with_mask(
//   CHECK-NOT:   iree_linalg_ext.map_scatter
//       CHECK:   vector.scatter
