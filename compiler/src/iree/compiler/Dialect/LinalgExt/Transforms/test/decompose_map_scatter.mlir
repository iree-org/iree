// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-map-scatter,canonicalize,cse))" \
// RUN:   --split-input-file --verify-diagnostics %s | FileCheck --check-prefix=CHECK %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-map-scatter{test-preprocessing-patterns=true},canonicalize,cse))" \
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
//   CHECK-DAG:   %[[CST:.+]] = arith.constant dense<16> : vector<4x1xindex>
//   CHECK-DAG:   %[[FLAT_OUTPUT:.+]] = memref.collapse_shape %[[OUTPUT]] {{.*}} memref<4x16xf32> into memref<64xf32>
//       CHECK:   %[[EXTRACT_IDX_0:.+]] = vector.extract %{{.*}}[0, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_0:.+]] = vector.extract %[[INPUT]][0] : vector<16xf32> from vector<4x16xf32>
//       CHECK:   vector.store %[[EXTRACT_0]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_0]]] : memref<64xf32>, vector<16xf32>
//       CHECK:   %[[EXTRACT_IDX_1:.+]] = vector.extract %{{.*}}[1, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_1:.+]] = vector.extract %[[INPUT]][1] : vector<16xf32> from vector<4x16xf32>
//       CHECK:   vector.store %[[EXTRACT_1]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_1]]] : memref<64xf32>, vector<16xf32>
//       CHECK:   %[[EXTRACT_IDX_2:.+]] = vector.extract %{{.*}}[2, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_2:.+]] = vector.extract %[[INPUT]][2] : vector<16xf32> from vector<4x16xf32>
//       CHECK:   vector.store %[[EXTRACT_2]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_2]]] : memref<64xf32>, vector<16xf32>
//       CHECK:   %[[EXTRACT_IDX_3:.+]] = vector.extract %{{.*}}[3, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_3:.+]] = vector.extract %[[INPUT]][3] : vector<16xf32> from vector<4x16xf32>
//       CHECK:   vector.store %[[EXTRACT_3]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_3]]] : memref<64xf32>, vector<16xf32>

// -----

// This test checks all index and mask computations for the `map_scatter` to `vector.scatter` path.
// Other tests shouldn't check this to avoid maintenance burden.

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
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[MASK:.+]] = arith.constant dense<true> : vector<256xi1>
//   CHECK-DAG:   %[[CST_64:.+]] = arith.constant dense<64> : vector<2x2x64xindex>
//   CHECK-DAG:   %[[CST_0:.+]] = arith.constant dense<0> : vector<64xindex>
//   CHECK-DAG:   %[[CST_2:.+]] = arith.constant dense<2> : vector<64xindex>
//   CHECK-DAG:   %[[CST_2_3D:.+]] = arith.constant dense<2> : vector<2x2x64xindex>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[FLAT_OUTPUT:.+]] = memref.collapse_shape %[[OUTPUT]] {{.*}} : memref<4x32x2xf32> into memref<256xf32>
//   CHECK-DAG:   %[[STEP_2:.+]] = vector.step : vector<2xindex>
//   CHECK-DAG:   %[[BROADCAST_64x2x2:.+]] = vector.broadcast %[[STEP_2]] : vector<2xindex> to vector<64x2x2xindex>
//   CHECK-DAG:   %[[TRANSPOSE_1:.+]] = vector.transpose %[[BROADCAST_64x2x2]], [2, 1, 0] : vector<64x2x2xindex> to vector<2x2x64xindex>
//   CHECK-DAG:   %[[BROADCAST_2x64x2:.+]] = vector.broadcast %[[STEP_2]] : vector<2xindex> to vector<2x64x2xindex>
//   CHECK-DAG:   %[[TRANSPOSE_2:.+]] = vector.transpose %[[BROADCAST_2x64x2]], [0, 2, 1] : vector<2x64x2xindex> to vector<2x2x64xindex>
//   CHECK-DAG:   %[[STEP_64:.+]] = vector.step : vector<64xindex>
//   CHECK-DAG:   %[[MULI_1:.+]] = arith.muli %[[TRANSPOSE_1]], %[[CST_2_3D]] overflow<nsw> : vector<2x2x64xindex>
//   CHECK-DAG:   %[[ADDI_1:.+]] = arith.addi %[[MULI_1]], %[[TRANSPOSE_2]] overflow<nsw> : vector<2x2x64xindex>
//   CHECK-DAG:   %[[FLOORDIVSI:.+]] = arith.floordivsi %[[STEP_64]], %[[CST_2]] : vector<64xindex>
//   CHECK-DAG:   %[[REMSI:.+]] = arith.remsi %[[STEP_64]], %[[CST_2]] : vector<64xindex>
//   CHECK-DAG:   %[[CMPI:.+]] = arith.cmpi slt, %[[REMSI]], %[[CST_0]] : vector<64xindex>
//   CHECK-DAG:   %[[ADDI_2:.+]] = arith.addi %[[REMSI]], %[[CST_2]] overflow<nsw> : vector<64xindex>
//   CHECK-DAG:   %[[SELECT:.+]] = arith.select %[[CMPI]], %[[ADDI_2]], %[[REMSI]] : vector<64xi1>, vector<64xindex>
//   CHECK-DAG:   %[[MULI_2:.+]] = arith.muli %[[ADDI_1]], %[[CST_64]] overflow<nsw> : vector<2x2x64xindex>
//   CHECK-DAG:   %[[MULI_3:.+]] = arith.muli %[[FLOORDIVSI]], %[[CST_2]] overflow<nsw> : vector<64xindex>
//   CHECK-DAG:   %[[BROADCAST_FINAL:.+]] = vector.broadcast %[[MULI_3]] : vector<64xindex> to vector<2x2x64xindex>
//   CHECK-DAG:   %[[ADDI_3:.+]] = arith.addi %[[MULI_2]], %[[BROADCAST_FINAL]] overflow<nsw> : vector<2x2x64xindex>
//       CHECK:   %[[BROADCAST_SELECT:.+]] = vector.broadcast %[[SELECT]] : vector<64xindex> to vector<2x2x64xindex>
//       CHECK:   %[[ADDI_FINAL:.+]] = arith.addi %[[ADDI_3]], %[[BROADCAST_SELECT]] overflow<nsw> : vector<2x2x64xindex>
//       CHECK:   %[[SHAPE_CAST_IDX:.+]] = vector.shape_cast %[[ADDI_FINAL]] : vector<2x2x64xindex> to vector<256xindex>
//       CHECK:   %[[SHAPE_CAST_DATA:.+]] = vector.shape_cast %[[INPUT]] : vector<2x2x64xf32> to vector<256xf32>
//       CHECK:   vector.scatter %[[FLAT_OUTPUT]][%[[C0]]] [%[[SHAPE_CAST_IDX]]], %[[MASK]], %[[SHAPE_CAST_DATA]] : memref<256xf32>, vector<256xindex>, vector<256xi1>, vector<256xf32>

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
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[DIM:.+]] = memref.dim %[[OUTPUT]], %[[C0]] : memref<?xf32>
//   CHECK-DAG:   %[[STEP:.+]] = vector.step : vector<64xindex>
//       CHECK:   %[[BROADCAST_DIM:.+]] = vector.broadcast %[[DIM]] : index to vector<64xindex>
//       CHECK:   %[[CMPI:.+]] = arith.cmpi uge, %[[STEP]], %[[BROADCAST_DIM]] : vector<64xindex>
//   CHECK-NOT:   iree_linalg_ext.map_scatter
//       CHECK:   vector.maskedstore %[[OUTPUT]][%[[C0]]], %[[CMPI]], %[[INPUT]] : memref<?xf32>, vector<64xi1>, vector<64xf32>

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
//   CHECK-DAG:   %[[FLAT_MASK:.+]] = arith.constant dense<true> : vector<64xi1>
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
//   CHECK-DAG:   %[[CST:.+]] = arith.constant dense<32> : vector<4x1xindex>
//   CHECK-DAG:   %[[SUBVIEW:.+]] = memref.subview %[[OUTPUT]]
//   CHECK-DAG:   %[[FLAT_OUTPUT:.+]] = memref.collapse_shape %[[SUBVIEW]] {{.*}} memref<4x32xf32{{.*}} into memref<128xf32
//       CHECK:   %[[EXTRACT_IDX_0:.+]] = vector.extract %{{.*}}[0, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_0:.+]] = vector.extract %[[INPUT]][0] : vector<16xf32> from vector<4x16xf32>
//       CHECK:   vector.store %[[EXTRACT_0]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_0]]] : memref<128xf32, strided<[1]>>, vector<16xf32>
//       CHECK:   %[[EXTRACT_IDX_1:.+]] = vector.extract %{{.*}}[1, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_1:.+]] = vector.extract %[[INPUT]][1] : vector<16xf32> from vector<4x16xf32>
//       CHECK:   vector.store %[[EXTRACT_1]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_1]]] : memref<128xf32, strided<[1]>>, vector<16xf32>
//       CHECK:   %[[EXTRACT_IDX_2:.+]] = vector.extract %{{.*}}[2, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_2:.+]] = vector.extract %[[INPUT]][2] : vector<16xf32> from vector<4x16xf32>
//       CHECK:   vector.store %[[EXTRACT_2]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_2]]] : memref<128xf32, strided<[1]>>, vector<16xf32>
//       CHECK:   %[[EXTRACT_IDX_3:.+]] = vector.extract %{{.*}}[3, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_3:.+]] = vector.extract %[[INPUT]][3] : vector<16xf32> from vector<4x16xf32>
//       CHECK:   vector.store %[[EXTRACT_3]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_3]]] : memref<128xf32, strided<[1]>>, vector<16xf32>

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
//   CHECK-DAG:   %{{.*}}, %[[OFFSET:.+]], %{{.*}}:2, %{{.*}}:2 = memref.extract_strided_metadata %[[OUTPUT]]
//   CHECK-DAG:   %[[FLAT_SIZE:.+]] = affine.apply #[[$MAP]]()[%[[D0]], %[[D1]]]
//       CHECK:   %[[FLAT_OUTPUT:.+]] = memref.reinterpret_cast %[[OUTPUT]]
//  CHECK-SAME:     to offset: [%[[OFFSET]]], sizes: [%[[FLAT_SIZE]]], strides: [1]
//  CHECK-SAME:     : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
//       CHECK:   %[[EXTRACT_IDX_0:.+]] = vector.extract %{{.*}}[0, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_0:.+]] = vector.extract %[[INPUT]][0] : vector<16xf32> from vector<4x16xf32>
//       CHECK:   vector.store %[[EXTRACT_0]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_0]]] : memref<?xf32, strided<[1], offset: ?>>, vector<16xf32>
//       CHECK:   %[[EXTRACT_IDX_1:.+]] = vector.extract %{{.*}}[1, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_1:.+]] = vector.extract %[[INPUT]][1] : vector<16xf32> from vector<4x16xf32>
//       CHECK:   vector.store %[[EXTRACT_1]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_1]]] : memref<?xf32, strided<[1], offset: ?>>, vector<16xf32>
//       CHECK:   %[[EXTRACT_IDX_2:.+]] = vector.extract %{{.*}}[2, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_2:.+]] = vector.extract %[[INPUT]][2] : vector<16xf32> from vector<4x16xf32>
//       CHECK:   vector.store %[[EXTRACT_2]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_2]]] : memref<?xf32, strided<[1], offset: ?>>, vector<16xf32>
//       CHECK:   %[[EXTRACT_IDX_3:.+]] = vector.extract %{{.*}}[3, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_3:.+]] = vector.extract %[[INPUT]][3] : vector<16xf32> from vector<4x16xf32>
//       CHECK:   vector.store %[[EXTRACT_3]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_3]]] : memref<?xf32, strided<[1], offset: ?>>, vector<16xf32>

// -----

func.func @map_scatter_sub_byte(
    %input: vector<4x16xf4E2M1FN>, %output: memref<4x16xf4E2M1FN, strided<[16, 1], offset: ?>>
) {
  iree_linalg_ext.map_scatter %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : vector<4x16xf4E2M1FN> into memref<4x16xf4E2M1FN, strided<[16, 1], offset: ?>>
  return
}
// CHECK-LABEL: func.func @map_scatter_sub_byte
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[FLAT_OUTPUT:.+]] = memref.collapse_shape %[[OUTPUT]] {{.*}} memref<4x16xf4E2M1FN{{.*}} into memref<64xf4E2M1FN{{.*}}
//       CHECK:   %[[EXTRACT_IDX_0:.+]] = vector.extract %{{.*}}[0, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_0:.+]] = vector.extract %[[INPUT]][0] : vector<16xf4E2M1FN> from vector<4x16xf4E2M1FN>
//       CHECK:   vector.store %[[EXTRACT_0]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_0]]] : memref<64xf4E2M1FN, strided<[1], offset: ?>>, vector<16xf4E2M1FN>
//       CHECK:   %[[EXTRACT_IDX_1:.+]] = vector.extract %{{.*}}[1, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_1:.+]] = vector.extract %[[INPUT]][1] : vector<16xf4E2M1FN> from vector<4x16xf4E2M1FN>
//       CHECK:   vector.store %[[EXTRACT_1]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_1]]] : memref<64xf4E2M1FN, strided<[1], offset: ?>>, vector<16xf4E2M1FN>
//       CHECK:   %[[EXTRACT_IDX_2:.+]] = vector.extract %{{.*}}[2, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_2:.+]] = vector.extract %[[INPUT]][2] : vector<16xf4E2M1FN> from vector<4x16xf4E2M1FN>
//       CHECK:   vector.store %[[EXTRACT_2]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_2]]] : memref<64xf4E2M1FN, strided<[1], offset: ?>>, vector<16xf4E2M1FN>
//       CHECK:   %[[EXTRACT_IDX_3:.+]] = vector.extract %{{.*}}[3, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_3:.+]] = vector.extract %[[INPUT]][3] : vector<16xf4E2M1FN> from vector<4x16xf4E2M1FN>
//       CHECK:   vector.store %[[EXTRACT_3]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_3]]] : memref<64xf4E2M1FN, strided<[1], offset: ?>>, vector<16xf4E2M1FN>

// -----

// This test checks all index and mask computations for the `map_scatter` to `vector.extract`/`vector.store` path.
// Other tests shouldn't check this to avoid maintenance burden.

func.func @map_scatter_sub_byte_with_mask(
    %input: vector<4x16xf4E2M1FN>, %output: memref<8x16xf4E2M1FN>
) {
  %c2 = arith.constant 2 : index
  iree_linalg_ext.map_scatter %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.cmpi ult, %idx0, %c2 : index
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : vector<4x16xf4E2M1FN> into memref<8x16xf4E2M1FN>
  return
}
// CHECK-LABEL: func.func @map_scatter_sub_byte_with_mask(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[CST_16:.+]] = arith.constant dense<16> : vector<4x1xindex>
//   CHECK-DAG:   %[[CST_2:.+]] = arith.constant dense<2> : vector<4x1xindex>
//   CHECK-DAG:   %[[FLAT_OUTPUT:.+]] = memref.collapse_shape %[[OUTPUT]] {{.*}} memref<8x16xf4E2M1FN> into memref<128xf4E2M1FN>
//   CHECK-DAG:   %[[STEP_4:.+]] = vector.step : vector<4xindex>
//   CHECK-DAG:   %[[BROADCAST_1x4:.+]] = vector.broadcast %[[STEP_4]] : vector<4xindex> to vector<1x4xindex>
//   CHECK-DAG:   %[[TRANSPOSE:.+]] = vector.transpose %[[BROADCAST_1x4]], [1, 0] : vector<1x4xindex> to vector<4x1xindex>
//   CHECK-DAG:   %[[STEP_1:.+]] = vector.step : vector<1xindex>
//   CHECK-DAG:   %[[CMPI:.+]] = arith.cmpi ult, %[[TRANSPOSE]], %[[CST_2]] : vector<4x1xindex>
//   CHECK-DAG:   %[[MULI:.+]] = arith.muli %[[TRANSPOSE]], %[[CST_16]] overflow<nsw> : vector<4x1xindex>
//   CHECK-DAG:   %[[BROADCAST_4x1:.+]] = vector.broadcast %[[STEP_1]] : vector<1xindex> to vector<4x1xindex>
//   CHECK-DAG:   %[[ADDI:.+]] = arith.addi %[[MULI]], %[[BROADCAST_4x1]] overflow<nsw> : vector<4x1xindex>
//       CHECK:   %[[EXTRACT_COND_0:.+]] = vector.extract %[[CMPI]][0, 0] : i1 from vector<4x1xi1>
//       CHECK:   %[[EXTRACT_IDX_0:.+]] = vector.extract %[[ADDI]][0, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_DATA_0:.+]] = vector.extract %[[INPUT]][0] : vector<16xf4E2M1FN> from vector<4x16xf4E2M1FN>
//       CHECK:   scf.if %[[EXTRACT_COND_0]] {
//       CHECK:     vector.store %[[EXTRACT_DATA_0]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_0]]] : memref<128xf4E2M1FN>, vector<16xf4E2M1FN>
//       CHECK:   }
//       CHECK:   %[[EXTRACT_COND_1:.+]] = vector.extract %[[CMPI]][1, 0] : i1 from vector<4x1xi1>
//       CHECK:   %[[EXTRACT_IDX_1:.+]] = vector.extract %[[ADDI]][1, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_DATA_1:.+]] = vector.extract %[[INPUT]][1] : vector<16xf4E2M1FN> from vector<4x16xf4E2M1FN>
//       CHECK:   scf.if %[[EXTRACT_COND_1]] {
//       CHECK:     vector.store %[[EXTRACT_DATA_1]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_1]]] : memref<128xf4E2M1FN>, vector<16xf4E2M1FN>
//       CHECK:   }
//       CHECK:   %[[EXTRACT_COND_2:.+]] = vector.extract %[[CMPI]][2, 0] : i1 from vector<4x1xi1>
//       CHECK:   %[[EXTRACT_IDX_2:.+]] = vector.extract %[[ADDI]][2, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_DATA_2:.+]] = vector.extract %[[INPUT]][2] : vector<16xf4E2M1FN> from vector<4x16xf4E2M1FN>
//       CHECK:   scf.if %[[EXTRACT_COND_2]] {
//       CHECK:     vector.store %[[EXTRACT_DATA_2]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_2]]] : memref<128xf4E2M1FN>, vector<16xf4E2M1FN>
//       CHECK:   }
//       CHECK:   %[[EXTRACT_COND_3:.+]] = vector.extract %[[CMPI]][3, 0] : i1 from vector<4x1xi1>
//       CHECK:   %[[EXTRACT_IDX_3:.+]] = vector.extract %[[ADDI]][3, 0] : index from vector<4x1xindex>
//       CHECK:   %[[EXTRACT_DATA_3:.+]] = vector.extract %[[INPUT]][3] : vector<16xf4E2M1FN> from vector<4x16xf4E2M1FN>
//   CHECK-NOT:   iree_linalg_ext.map_scatter
//       CHECK:   scf.if %[[EXTRACT_COND_3]] {
//       CHECK:     vector.store %[[EXTRACT_DATA_3]], %[[FLAT_OUTPUT]][%[[EXTRACT_IDX_3]]] : memref<128xf4E2M1FN>, vector<16xf4E2M1FN>
//       CHECK:   }

// -----

func.func @map_scatter_sub_byte_not_unit_stride(
    %input: vector<2x2xf4E2M1FN>, %output: memref<2x4xf4E2M1FN>
) {
  // expected-error@+1 {{with an access on a sub-byte type that is not a multiple of the byte size can't be vectorized}}
  iree_linalg_ext.map_scatter %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0) -> (d0 * 2)>(%idx1)
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : vector<2x2xf4E2M1FN> into memref<2x4xf4E2M1FN>
  return
}

// -----

func.func @map_scatter_with_mask_on_inner_dim(
    %input: vector<4x16xf4E2M1FN>, %output: memref<8x16xf4E2M1FN>
) {
  %c4 = arith.constant 4 : index
  // expected-error@+1 {{on sub-byte type with potentially non byte aligned transformation}}
  iree_linalg_ext.map_scatter %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.cmpi ult, %idx1, %c4 : index
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : vector<4x16xf4E2M1FN> into memref<8x16xf4E2M1FN>
  return
}
