// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{enable-vector-masking=false vectorize-map-store=true}))" --split-input-file %s | FileCheck %s

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
// CHECK-LABEL: @map_store
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_store
//  CHECK-SAME:     %[[READ]] into %[[OUTPUT]]
//       CHECK:     : vector<4x16x64xf32> into tensor<4x16x64xf32> -> tensor<4x16x64xf32>
//       CHECK:   return %[[MAP_SCATTER]] : tensor<4x16x64xf32>

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
// CHECK-LABEL: @no_vectorize_map_store_dynamic
//   CHECK-NOT:   vector

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
// CHECK-LABEL: @map_store_f4_multiple_of_byte
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_store
//  CHECK-SAME:     %[[READ]] into %[[OUTPUT]]
//       CHECK:     : vector<2x2xf4E2M1FN> into tensor<2x2xf4E2M1FN> -> tensor<2x2xf4E2M1FN>
//       CHECK:   return %[[MAP_SCATTER]] : tensor<2x2xf4E2M1FN>

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
// CHECK-LABEL: @map_store_f4_not_multiple_of_byte
//   CHECK-NOT:   vector

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
// CHECK-LABEL: @map_store_f4_unit_stride
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_store
//  CHECK-SAME:     %[[READ]] into %[[OUTPUT]]
//       CHECK:     : vector<2x2xf4E2M1FN> into tensor<2x4xf4E2M1FN> -> tensor<2x4xf4E2M1FN>
//       CHECK:   return %[[MAP_SCATTER]] : tensor<2x4xf4E2M1FN>

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
// CHECK-LABEL: @map_store_f4_not_unit_stride
//   CHECK-NOT:   vector

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
// CHECK-LABEL: @map_store_f4_not_index_applied_multiple_times
//   CHECK-NOT:   vector

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
// CHECK-LABEL: @map_store_f4_mask_depends_on_inner_index
//   CHECK-NOT:   vector

// -----

// A `map_store` whose body uses the per-group `affine.linearize_index disjoint`
// form -- the IR shape `foldReshapeIntoMapStore` emits when folding a
// `collapse_shape` per reassociation group rather than via one global linearize
// + delinearize. Vectorization accepts it when the input is statically shaped
// (the relevant criterion is `inputType.hasStaticShape()`, the index
// transformation in the body is opaque to that check).
func.func @vectorize_map_store_per_group_linearize(
    %input: tensor<2x4x16xf32>, %output: tensor<8x16xf32>
) -> tensor<8x16xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index, %idx2: index):
      %lin = affine.linearize_index disjoint [%idx0, %idx1] by (2, 4) : index
      %mask = arith.constant true
      iree_linalg_ext.yield %lin, %idx2, %mask : index, index, i1
  } : tensor<2x4x16xf32> into tensor<8x16xf32> -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
// CHECK-LABEL: @vectorize_map_store_per_group_linearize
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_store
//  CHECK-SAME:     %[[READ]] into %[[OUTPUT]]
//       CHECK:     : vector<2x4x16xf32> into tensor<8x16xf32> -> tensor<8x16xf32>

// -----

// Expand-direction companion of the above: per-group `affine.delinearize_index`
// in the body (the shape emitted when folding an `expand_shape` per
// reassociation group). Same story -- vectorization is gated on the input's
// static shape, not on what the index transformation looks like.
func.func @vectorize_map_store_per_group_delinearize(
    %input: tensor<8x16xf32>, %output: tensor<2x4x16xf32>
) -> tensor<2x4x16xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %delin:2 = affine.delinearize_index %idx0 into (2, 4) : index, index
      %mask = arith.constant true
      iree_linalg_ext.yield %delin#0, %delin#1, %idx1, %mask : index, index, index, i1
  } : tensor<8x16xf32> into tensor<2x4x16xf32> -> tensor<2x4x16xf32>
  return %0 : tensor<2x4x16xf32>
}
// CHECK-LABEL: @vectorize_map_store_per_group_delinearize
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_store
//  CHECK-SAME:     %[[READ]] into %[[OUTPUT]]
//       CHECK:     : vector<8x16xf32> into tensor<2x4x16xf32> -> tensor<2x4x16xf32>

// -----

// Dynamic-input counterpart of @vectorize_map_store_per_group_delinearize:
// the per-group form is what reaches this point in practice for CPU-tiled
// encoding relayouts, but the vectorizer still skips it because the input is
// dynamically shaped -- consistent with @no_vectorize_map_store_dynamic above.
// The per-group fold's payoff is at the later scalarization stage (static
// shifts/masks instead of dynamic integer division per element), not at
// vectorization.
func.func @no_vectorize_map_store_per_group_delinearize_dynamic(
    %input: tensor<?xf32>, %output: tensor<?x4xf32>, %d0 : index
) -> tensor<?x4xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index):
      %delin:2 = affine.delinearize_index %idx0 into (%d0, 4) : index, index
      %mask = arith.constant true
      iree_linalg_ext.yield %delin#0, %delin#1, %mask : index, index, i1
  } : tensor<?xf32> into tensor<?x4xf32> -> tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}
// CHECK-LABEL: @no_vectorize_map_store_per_group_delinearize_dynamic
//   CHECK-NOT:   vector
