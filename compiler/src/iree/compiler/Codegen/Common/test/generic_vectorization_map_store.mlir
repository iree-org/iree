// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{vectorize-map-store=true}))" --split-input-file %s | FileCheck %s --check-prefix=CHECK-MAP-STORE

func.func @map_store(
    %input: tensor<4x16x64xf32>, %output: tensor<4x16x64xf32>
) -> tensor<4x16x64xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index, %idx2: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %idx2, %mask : index, index, index, i1
  } : tensor<4x16x64xf32> into tensor<4x16x64xf32> -> tensor<4x16x64xf32>
  return %0 : tensor<4x16x64xf32>
}
// CHECK-MAP-STORE-LABEL: @map_store
//  CHECK-MAP-STORE-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-MAP-STORE-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK-MAP-STORE:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK-MAP-STORE:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_store
//  CHECK-MAP-STORE-SAME:     %[[READ]] into %[[OUTPUT]]
//       CHECK-MAP-STORE:     : vector<4x16x64xf32> into tensor<4x16x64xf32> -> tensor<4x16x64xf32>
//       CHECK-MAP-STORE:   return %[[MAP_SCATTER]] : tensor<4x16x64xf32>

// -----

func.func @no_vectorize_map_store_dynamic(
    %input: tensor<?xf32>, %output: tensor<64xf32>
) -> tensor<64xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %mask : index, i1
  } : tensor<?xf32> into tensor<64xf32> -> tensor<64xf32>
  return %0 : tensor<64xf32>
}
// CHECK-MAP-STORE-LABEL: @no_vectorize_map_store_dynamic
//   CHECK-MAP-STORE-NOT:   vector

// -----

func.func @map_store_f4_multiple_of_byte(
    %input: tensor<2x2xf4E2M1FN>, %output: tensor<2x2xf4E2M1FN>
) -> tensor<2x2xf4E2M1FN> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : tensor<2x2xf4E2M1FN> into tensor<2x2xf4E2M1FN> -> tensor<2x2xf4E2M1FN>
  return %0 : tensor<2x2xf4E2M1FN>
}
// CHECK-MAP-STORE-LABEL: @map_store_f4_multiple_of_byte
//  CHECK-MAP-STORE-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-MAP-STORE-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK-MAP-STORE:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK-MAP-STORE:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_store
//  CHECK-MAP-STORE-SAME:     %[[READ]] into %[[OUTPUT]]
//       CHECK-MAP-STORE:     : vector<2x2xf4E2M1FN> into tensor<2x2xf4E2M1FN> -> tensor<2x2xf4E2M1FN>
//       CHECK-MAP-STORE:   return %[[MAP_SCATTER]] : tensor<2x2xf4E2M1FN>

// -----

func.func @map_store_f4_not_multiple_of_byte(
    %input: tensor<2x1xf4E2M1FN>, %output: tensor<2x2xf4E2M1FN>
) -> tensor<2x2xf4E2M1FN> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : tensor<2x1xf4E2M1FN> into tensor<2x2xf4E2M1FN> -> tensor<2x2xf4E2M1FN>
  return %0 : tensor<2x2xf4E2M1FN>
}
// CHECK-MAP-STORE-LABEL: @map_store_f4_not_multiple_of_byte
//   CHECK-MAP-STORE-NOT:   vector

// -----

func.func @map_store_f4_unit_stride(
    %input: tensor<2x2xf4E2M1FN>, %output: tensor<2x4xf4E2M1FN>
) -> tensor<2x4xf4E2M1FN> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0) -> (d0 + 2)>(%idx1)
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x2xf4E2M1FN> into tensor<2x4xf4E2M1FN> -> tensor<2x4xf4E2M1FN>
  return %0 : tensor<2x4xf4E2M1FN>
}
// CHECK-MAP-STORE-LABEL: @map_store_f4_unit_stride
//  CHECK-MAP-STORE-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-MAP-STORE-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK-MAP-STORE:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK-MAP-STORE:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_store
//  CHECK-MAP-STORE-SAME:     %[[READ]] into %[[OUTPUT]]
//       CHECK-MAP-STORE:     : vector<2x2xf4E2M1FN> into tensor<2x4xf4E2M1FN> -> tensor<2x4xf4E2M1FN>
//       CHECK-MAP-STORE:   return %[[MAP_SCATTER]] : tensor<2x4xf4E2M1FN>

// -----

func.func @map_store_f4_not_unit_stride(
    %input: tensor<2x2xf4E2M1FN>, %output: tensor<2x4xf4E2M1FN>
) -> tensor<2x4xf4E2M1FN> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0) -> (d0 * 2)>(%idx1)
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x2xf4E2M1FN> into tensor<2x4xf4E2M1FN> -> tensor<2x4xf4E2M1FN>
  return %0 : tensor<2x4xf4E2M1FN>
}
// CHECK-MAP-STORE-LABEL: @map_store_f4_not_unit_stride
//   CHECK-MAP-STORE-NOT:   vector

// -----

func.func @map_store_f4_not_index_applied_multiple_times(
    %input: tensor<2x2xf4E2M1FN>, %output: tensor<2x4xf4E2M1FN>
) -> tensor<2x4xf4E2M1FN> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%idx1, %idx1)
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x2xf4E2M1FN> into tensor<2x4xf4E2M1FN> -> tensor<2x4xf4E2M1FN>
  return %0 : tensor<2x4xf4E2M1FN>
}
// CHECK-MAP-STORE-LABEL: @map_store_f4_not_index_applied_multiple_times
//   CHECK-MAP-STORE-NOT:   vector

// -----

func.func @map_store_f4_mask_depends_on_inner_index(
    %input: tensor<2x2xf4E2M1FN>, %output: tensor<2x4xf4E2M1FN>
) -> tensor<2x4xf4E2M1FN> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %c1 = arith.constant 1 : index
      %mask = arith.cmpi uge, %idx1, %c1 : index
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : tensor<2x2xf4E2M1FN> into tensor<2x4xf4E2M1FN> -> tensor<2x4xf4E2M1FN>
  return %0 : tensor<2x4xf4E2M1FN>
}
// CHECK-MAP-STORE-LABEL: @map_store_f4_mask_depends_on_inner_index
//   CHECK-MAP-STORE-NOT:   vector
