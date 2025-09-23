// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-map-scatter,cse))" \
// RUN:   --split-input-file %s | FileCheck --check-prefix=CHECK %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-map-scatter{test-preprocessing-patterns=true},cse))" \
// RUN:   --split-input-file %s | FileCheck --check-prefix=PREPROCESSING %s

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

// -----

func.func @map_scatter_into_subview(
    %input: vector<4x16xf32>, %output: memref<8x32xf32>
) {
  %subview = memref.subview %output[2, 7][4, 16][1, 1] : memref<8x32xf32> to memref<4x16xf32, strided<[32, 1], offset: 71>>
  iree_linalg_ext.map_scatter %input into %subview {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : vector<4x16xf32> into memref<4x16xf32, strided<[32, 1], offset: 71>>
  return
}
// CHECK-LABEL: func.func @map_scatter_into_subview(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//   CHECK-NOT:   memref.subview
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[FLAT_OUTPUT:.+]] = memref.collapse_shape %[[OUTPUT]] {{.*}} memref<8x32xf32> into memref<256xf32>
//   CHECK-DAG:   %[[FLAT_INDICES:.+]] = vector.shape_cast{{.*}} : vector<4x16xindex> to vector<64xindex>
//   CHECK-DAG:   %[[FLAT_MASK:.+]] = vector.shape_cast{{.*}} : vector<4x16xi1> to vector<64xi1>
//   CHECK-DAG:   %[[FLAT_INPUT:.+]] = vector.shape_cast %[[INPUT]] : vector<4x16xf32> to vector<64xf32>
//       CHECK:   vector.scatter %[[FLAT_OUTPUT]][%[[C0]]]
//  CHECK-SAME:     [%[[FLAT_INDICES]]], %[[FLAT_MASK]], %[[FLAT_INPUT]]

// PREPROCESSING-LABEL: func.func @map_scatter_into_subview(
//  PREPROCESSING-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  PREPROCESSING-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//   PREPROCESSING-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   PREPROCESSING-DAG:   %[[C7:.+]] = arith.constant 7 : index
//   PREPROCESSING-NOT:   memref.subview
//       PREPROCESSING:   iree_linalg_ext.map_scatter %[[INPUT]] into %[[OUTPUT]] {
//       PREPROCESSING:     ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//       PREPROCESSING:       %[[OUT_IDX0:.+]] = arith.addi %[[IDX0]], %[[C2]]
//       PREPROCESSING:       %[[OUT_IDX1:.+]] = arith.addi %[[IDX1]], %[[C7]]
//       PREPROCESSING:       iree_linalg_ext.yield %[[OUT_IDX0]], %[[OUT_IDX1]]
//       PREPROCESSING:   } : vector<4x16xf32> into memref<8x32xf32>

// -----

func.func @map_scatter_into_collapsible_subview(
    %input: vector<4x16xf32>, %output: memref<8x32xf32>
) {
    %subview = memref.subview %output[0, 0][4, 32][1, 1] : memref<8x32xf32> to memref<4x32xf32, strided<[32, 1]>>
    iree_linalg_ext.map_scatter %input into %subview {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : vector<4x16xf32> into memref<4x32xf32, strided<[32, 1]>>
  return
}
// CHECK-LABEL: func.func @map_scatter_into_collapsible_subview(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[SUBVIEW:.+]] = memref.subview %[[OUTPUT]]
//   CHECK-DAG:   %[[FLAT_OUTPUT:.+]] = memref.collapse_shape %[[SUBVIEW]] {{.*}} memref<4x32xf32{{.*}} into memref<128xf32
//   CHECK-DAG:   %[[FLAT_INDICES:.+]] = vector.shape_cast{{.*}} : vector<4x16xindex> to vector<64xindex>
//   CHECK-DAG:   %[[FLAT_MASK:.+]] = vector.shape_cast{{.*}} : vector<4x16xi1> to vector<64xi1>
//   CHECK-DAG:   %[[FLAT_INPUT:.+]] = vector.shape_cast %[[INPUT]] : vector<4x16xf32> to vector<64xf32>
//       CHECK:   vector.scatter %[[FLAT_OUTPUT]][%[[C0]]]
//  CHECK-SAME:     [%[[FLAT_INDICES]]], %[[FLAT_MASK]], %[[FLAT_INPUT]]

// PREPROCESSING-LABEL: func.func @map_scatter_into_collapsible_subview(
//  PREPROCESSING:        memref.subview

// -----

func.func @map_scatter_into_strided_output(
    %input: vector<4x16xf32>, %output: memref<?x?xf32, strided<[?, ?], offset: ?>>
) {
  iree_linalg_ext.map_scatter %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : vector<4x16xf32> into memref<?x?xf32, strided<[?, ?], offset: ?>>
  return
}
//       CHECK: #[[$MAP:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK-LABEL: func.func @map_scatter_into_strided_output(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = memref.dim %[[OUTPUT]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = memref.dim %[[OUTPUT]], %[[C1]]
//   CHECK-DAG:   %{{.*}}, %[[OFFSET:.+]], %{{.*}}:2, %{{.*}} = memref.extract_strided_metadata %[[OUTPUT]]
//   CHECK-DAG:   %[[FLAT_SIZE:.+]] = affine.apply #[[$MAP]]()[%[[D0]], %[[D1]]]
//       CHECK:   %[[FLAT_OUTPUT:.+]] = memref.reinterpret_cast %[[OUTPUT]]
//  CHECK-SAME:     to offset: [%[[OFFSET]]], sizes: [%[[FLAT_SIZE]]], strides: [1]
//  CHECK-SAME:     : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
//   CHECK-DAG:   %[[FLAT_INDICES:.+]] = vector.shape_cast{{.*}} : vector<4x16xindex> to vector<64xindex>
//   CHECK-DAG:   %[[FLAT_MASK:.+]] = vector.shape_cast{{.*}} : vector<4x16xi1> to vector<64xi1>
//   CHECK-DAG:   %[[FLAT_INPUT:.+]] = vector.shape_cast %[[INPUT]] : vector<4x16xf32> to vector<64xf32>
//       CHECK:   vector.scatter %[[FLAT_OUTPUT]][%[[C0]]]
//  CHECK-SAME:     [%[[FLAT_INDICES]]], %[[FLAT_MASK]], %[[FLAT_INPUT]]
