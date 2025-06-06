// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-combine-layout-transformation,canonicalize,cse))" -split-input-file %s | FileCheck %s

func.func @fold_collapse_shape_op(%source : tensor<2x4x16xf32>, %result : memref<8x16xf32>) {
  %collapse = tensor.collapse_shape %source [[0, 1], [2]] : tensor<2x4x16xf32> into tensor<8x16xf32>
  iree_codegen.store_to_buffer %collapse, %result : tensor<8x16xf32> into memref<8x16xf32>
  return
}
// CHECK-LABEL: @fold_collapse_shape_op
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
//       CHECK:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<8x16xf32>
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  CHECK-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index):
//       CHECK:     %[[LINEARIZE:.+]] = affine.linearize_index
//  CHECK-SAME:       [%[[IDX0]], %[[IDX1]]] by (2, 4)
//       CHECK:     iree_linalg_ext.yield %[[LINEARIZE]], %[[IDX2]], %[[TRUE]]
//       CHECK:   } : tensor<2x4x16xf32> into tensor<8x16xf32> -> tensor<8x16xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<8x16xf32> into memref<8x16xf32>

// -----

func.func @fold_expand_shape_op(%source : tensor<8x16xf32>, %result : memref<2x4x16xf32>) {
  %expand = tensor.expand_shape %source [[0, 1], [2]] output_shape [2, 4, 16] : tensor<8x16xf32> into tensor<2x4x16xf32>
  iree_codegen.store_to_buffer %expand, %result : tensor<2x4x16xf32> into memref<2x4x16xf32>
  return
}
// CHECK-LABEL: @fold_expand_shape_op
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
//       CHECK:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<2x4x16xf32>
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  CHECK-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//       CHECK:     %[[DELINEARIZE:.+]]:2 = affine.delinearize_index %[[IDX0]] into (2, 4)
//       CHECK:     iree_linalg_ext.yield %[[DELINEARIZE]]#0, %[[DELINEARIZE]]#1, %[[IDX1]], %[[TRUE]]
//       CHECK:   } : tensor<8x16xf32> into tensor<2x4x16xf32> -> tensor<2x4x16xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<2x4x16xf32> into memref<2x4x16xf32>

// -----

func.func @fold_transpose_op(%source : tensor<2x4x16xf32>, %result : memref<4x16x2xf32>) {
  %init = tensor.empty() : tensor<4x16x2xf32>
  %transposed = linalg.transpose ins(%source : tensor<2x4x16xf32>) outs(%init : tensor<4x16x2xf32>) permutation = [1, 2, 0]
  iree_codegen.store_to_buffer %transposed, %result : tensor<4x16x2xf32> into memref<4x16x2xf32>
  return
}
// CHECK-LABEL: @fold_transpose_op
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
//       CHECK:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<4x16x2xf32>
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  CHECK-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index):
//       CHECK:     iree_linalg_ext.yield %[[IDX1]], %[[IDX2]], %[[IDX0]], %[[TRUE]]
//       CHECK:   } : tensor<2x4x16xf32> into tensor<4x16x2xf32> -> tensor<4x16x2xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<4x16x2xf32> into memref<4x16x2xf32>

// -----

func.func @fold_extract_slice_op(%source : tensor<64xf32>, %result : memref<63xf32>) {
  %slice = tensor.extract_slice %source[0] [63] [1] : tensor<64xf32> to tensor<63xf32>
  iree_codegen.store_to_buffer %slice, %result : tensor<63xf32> into memref<63xf32>
  return
}
// CHECK-LABEL: @fold_extract_slice_op
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C63:.+]] = arith.constant 63 : index
//       CHECK:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<63xf32>
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  CHECK-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index):
//       CHECK:     %[[MASK:.+]] = arith.cmpi ult, %[[IDX0]], %[[C63]]
//       CHECK:     iree_linalg_ext.yield %[[IDX0]], %[[MASK]]
//       CHECK:   } : tensor<64xf32> into tensor<63xf32> -> tensor<63xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<63xf32> into memref<63xf32>

// -----

func.func @no_fold_offset_extract_slice_op(%source : tensor<64xf32>, %result : memref<4xf32>) {
  %slice = tensor.extract_slice %source[42] [4] [1] : tensor<64xf32> to tensor<4xf32>
  iree_codegen.store_to_buffer %slice, %result : tensor<4xf32> into memref<4xf32>
  return
}
// CHECK-LABEL: @no_fold_offset_extract_slice_op
//       CHECK:   tensor.extract_slice
//   CHECK-NOT:   iree_linalg_ext.map_scatter

// -----

func.func @no_fold_strided_extract_slice_op(%source : tensor<64xf32>, %result : memref<16xf32>) {
  %slice = tensor.extract_slice %source[0] [16] [4] : tensor<64xf32> to tensor<16xf32>
  iree_codegen.store_to_buffer %slice, %result : tensor<16xf32> into memref<16xf32>
  return
}
// CHECK-LABEL: @no_fold_strided_extract_slice_op
//       CHECK:   tensor.extract_slice
//   CHECK-NOT:   iree_linalg_ext.map_scatter

// -----

func.func @fold_pad_op(%source : tensor<250xf32>, %result : memref<256xf32>) {
  %cst = arith.constant 0.0 : f32
  %padded = tensor.pad %source low[2] high[4] {
  ^bb0(%arg0: index):
    tensor.yield %cst : f32
  } : tensor<250xf32> to tensor<256xf32>
  iree_codegen.store_to_buffer %padded, %result : tensor<256xf32> into memref<256xf32>
  return
}
//       CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (256, d0 + 64)>
// CHECK-LABEL: @fold_pad_op
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[PAD_VAL:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C252:.+]] = arith.constant 252 : index
//       CHECK:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<256xf32>
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  CHECK-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index):
//       CHECK:     iree_linalg_ext.yield %[[IDX0]], %[[TRUE]]
//       CHECK:   } : tensor<250xf32> into tensor<256xf32> -> tensor<256xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<256xf32> into memref<256xf32>

//       CHECK:   scf.forall (%[[WG_IV:.+]]) = (0) to (256) step (64) {
//       CHECK:     %[[WG_TILE_UB:.+]] = affine.min #[[$MAP]](%[[WG_IV]])
//       CHECK:     scf.for %[[IDX:.+]] = %[[WG_IV]] to %[[WG_TILE_UB]] step %[[C1]] {
//   CHECK-DAG:       %[[IS_LOW_PAD:.+]] = arith.cmpi ult, %[[IDX]], %[[C2]]
//   CHECK-DAG:       %[[IS_HIGH_PAD:.+]] = arith.cmpi uge, %[[IDX]], %[[C252]]
//   CHECK-DAG:       %[[IS_PAD:.+]] = arith.ori %[[IS_LOW_PAD]], %[[IS_HIGH_PAD]] : i1
//       CHECK:       scf.if %[[IS_PAD]] {
//  CHECK-NEXT:         memref.store %[[PAD_VAL]], %[[RESULT]][%[[IDX]]] : memref<256xf32>
//  CHECK-NEXT:       }
//  CHECK:          }
//  CHECK:        } {mapping = [#iree_codegen.workgroup_mapping<x>]}

// -----

func.func @fold_unpack_op(%source : tensor<?x?x128x128xf32>, %result : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = memref.dim %result, %c0 : memref<?x?xf32>
  %d1 = memref.dim %result, %c1 : memref<?x?xf32>
  %dest = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %unpack = linalg.unpack %source
      outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [128, 128]
      into %dest : tensor<?x?x128x128xf32> -> tensor<?x?xf32>
  iree_codegen.store_to_buffer %unpack, %result : tensor<?x?xf32> into memref<?x?xf32>
  return
}
//       CHECK: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 * 128)>
// CHECK-LABEL: @fold_unpack_op
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[RES_D0:.+]] = memref.dim %[[RESULT]], %[[C0]] : memref<?x?xf32>
//   CHECK-DAG:   %[[RES_D1:.+]] = memref.dim %[[RESULT]], %[[C1]] : memref<?x?xf32>
//   CHECK-DAG:   %[[SRC_D0:.+]] = tensor.dim %[[SOURCE]], %[[C0]] : tensor<?x?x128x128xf32>
//   CHECK-DAG:   %[[SRC_D1:.+]] = tensor.dim %[[SOURCE]], %[[C1]] : tensor<?x?x128x128xf32>
//   CHECK-DAG:   %[[COLLAPSE_SIZE0:.+]] = affine.apply #[[$MAP]]()[%[[SRC_D0]]]
//   CHECK-DAG:   %[[COLLAPSE_SIZE1:.+]] = affine.apply #[[$MAP]]()[%[[SRC_D1]]]
//       CHECK:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty(%[[RES_D0]], %[[RES_D1]]) : tensor<?x?xf32>
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  CHECK-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index, %[[IDX3:.+]]: index):
//       CHECK:     %[[LINEARIZE:.+]] = affine.linearize_index
//  CHECK-SAME:       [%[[IDX0]], %[[IDX2]], %[[IDX1]], %[[IDX3]]]
//  CHECK-SAME:       by (%[[SRC_D0]], 128, %[[SRC_D1]], 128)
//       CHECK:     %[[DELINEARIZE:.+]]:2 = affine.delinearize_index %[[LINEARIZE]]
//  CHECK-SAME:       into (%[[COLLAPSE_SIZE0]], %[[COLLAPSE_SIZE1]])
//       CHECK:     %[[BOUND0:.+]] = arith.cmpi ult, %[[DELINEARIZE]]#0, %[[RES_D0]]
//       CHECK:     %[[BOUND1:.+]] = arith.cmpi ult, %[[DELINEARIZE]]#1, %[[RES_D1]]
//       CHECK:     %[[MASK:.+]] = arith.andi %[[BOUND0]], %[[BOUND1]] : i1
//       CHECK:     iree_linalg_ext.yield %[[DELINEARIZE]]#0, %[[DELINEARIZE]]#1, %[[MASK]]
//       CHECK:   } : tensor<?x?x128x128xf32> into tensor<?x?xf32> -> tensor<?x?xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<?x?xf32> into memref<?x?xf32>

// -----

func.func @fold_pack_op(%source : tensor<250x250xf32>, %result : memref<2x2x128x128xf32>) {
  %cst = arith.constant 0.0 : f32
  %dest = tensor.empty() : tensor<2x2x128x128xf32>
  %unpack = linalg.pack %source padding_value(%cst : f32)
      outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [128, 128]
      into %dest : tensor<250x250xf32> -> tensor<2x2x128x128xf32>
  iree_codegen.store_to_buffer %unpack, %result : tensor<2x2x128x128xf32> into memref<2x2x128x128xf32>
  return
}
//       CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (256, d0 + 1)>
//       CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (256, d0 + 64)>
// CHECK-LABEL: @fold_pack_op
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
//   CHECK-DAG:   %[[PAD_VAL:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C250:.+]] = arith.constant 250 : index
//       CHECK:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<2x2x128x128xf32>
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  CHECK-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//       CHECK:     %[[DELINEARIZE0:.+]]:2 = affine.delinearize_index %[[IDX0]]
//  CHECK-SAME:       into (2, 128)
//       CHECK:     %[[DELINEARIZE1:.+]]:2 = affine.delinearize_index %[[IDX1]]
//  CHECK-SAME:       into (2, 128)
//       CHECK:     iree_linalg_ext.yield %[[DELINEARIZE0]]#0, %[[DELINEARIZE1]]#0, %[[DELINEARIZE0]]#1, %[[DELINEARIZE1]]#1, %[[TRUE]]
//       CHECK:   } : tensor<250x250xf32> into tensor<2x2x128x128xf32> -> tensor<2x2x128x128xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<2x2x128x128xf32> into memref<2x2x128x128xf32>

//       CHECK:   scf.forall (%[[WG_IV0:.+]], %[[WG_IV1:.+]]) = (0, 0) to (256, 256) step (1, 64) {
//   CHECK-DAG:     %[[WG_TILE_UB0:.+]] = affine.min #[[$MAP]](%[[WG_IV0]])
//   CHECK-DAG:     %[[WG_TILE_UB1:.+]] = affine.min #[[$MAP1]](%[[WG_IV1]])
//       CHECK:     scf.for %[[IDX0:.+]] = %[[WG_IV0]] to %[[WG_TILE_UB0]] step %[[C1]] {
//       CHECK:       scf.for %[[IDX1:.+]] = %[[WG_IV1]] to %[[WG_TILE_UB1]] step %[[C1]] {
//   CHECK-DAG:         %[[EXPANDED_IDX0:.+]]:2 = affine.delinearize_index %[[IDX0]] into (2, 128)
//   CHECK-DAG:         %[[EXPANDED_IDX1:.+]]:2 = affine.delinearize_index %[[IDX1]] into (2, 128)
//   CHECK-DAG:         %[[IDX0_IS_LOW_PAD:.+]] = arith.cmpi ult, %[[IDX0]], %[[C0]]
//   CHECK-DAG:         %[[IDX0_IS_HIGH_PAD:.+]] = arith.cmpi uge, %[[IDX0]], %[[C250]]
//   CHECK-DAG:         %[[IDX0_IS_PAD:.+]] = arith.ori %[[IDX0_IS_LOW_PAD]], %[[IDX0_IS_HIGH_PAD]] : i1
//   CHECK-DAG:         %[[IDX1_IS_LOW_PAD:.+]] = arith.cmpi ult, %[[IDX1]], %[[C0]]
//   CHECK-DAG:         %[[IDX1_IS_HIGH_PAD:.+]] = arith.cmpi uge, %[[IDX1]], %[[C250]]
//   CHECK-DAG:         %[[IDX1_IS_PAD:.+]] = arith.ori %[[IDX1_IS_LOW_PAD]], %[[IDX1_IS_HIGH_PAD]] : i1
//   CHECK-DAG:         %[[IS_PAD:.+]] = arith.ori %[[IDX0_IS_PAD]], %[[IDX1_IS_PAD]] : i1
//       CHECK:         scf.if %[[IS_PAD]] {
//  CHECK-NEXT:           memref.store %[[PAD_VAL]], %[[RESULT]]
//  CHECK-SAME:             [%[[EXPANDED_IDX0]]#0, %[[EXPANDED_IDX1]]#0, %[[EXPANDED_IDX0]]#1, %[[EXPANDED_IDX1]]#1]
//  CHECK-SAME:             : memref<2x2x128x128xf32>
//  CHECK-NEXT:         }
//  CHECK:            }
//  CHECK:          }
//  CHECK:        } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

// -----

func.func @propagate_relayout_ops(%source : tensor<?x?x128x128xf32>,
                                  %result : memref<?xf16>,
                                  %size0: index, %size1: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dest = tensor.empty(%size0, %size1) : tensor<?x?xf32>
  %unpack = linalg.unpack %source
      outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [128, 128]
      into %dest : tensor<?x?x128x128xf32> -> tensor<?x?xf32>
  %collapse = tensor.collapse_shape %unpack [[0, 1]] : tensor<?x?xf32> into tensor<?xf32>
  %flat_size = arith.muli %size0, %size1 : index
  %init = tensor.empty(%flat_size) : tensor<?xf16>
  %compute_op = linalg.generic
      {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
       iterator_types = ["parallel"]}
       ins(%collapse : tensor<?xf32>) outs(%init : tensor<?xf16>) {
  ^bb0(%in: f32, %out: f16):
    %trunc = arith.truncf %in : f32 to f16
    linalg.yield %trunc : f16
  } -> tensor<?xf16>
  iree_codegen.store_to_buffer %compute_op, %result : tensor<?xf16> into memref<?xf16>
  return
}
// CHECK-LABEL: @propagate_relayout_ops
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//       CHECK:   %[[INIT:.+]] = tensor.empty{{.*}} : tensor<?x?x128x128xf16>
//       CHECK:   %[[COMPUTE_OP:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[SOURCE]] : tensor<?x?x128x128xf32>)
//  CHECK-SAME:     outs(%[[INIT]] : tensor<?x?x128x128xf16>)
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter %[[COMPUTE_OP]]
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]]
