// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-combine-layout-transformation{scope=dispatch},canonicalize,cse))" \
// RUN:   -split-input-file %s | FileCheck %s --check-prefixes=DISPATCH-SCOPE
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-combine-layout-transformation{scope=workgroup},canonicalize,cse))" \
// RUN:   -split-input-file %s | FileCheck %s --check-prefixes=WORKGROUP-SCOPE

func.func @fold_collapse_shape_op(%source : tensor<2x4x16xf32>, %result : memref<8x16xf32>) {
  %collapse = tensor.collapse_shape %source [[0, 1], [2]] : tensor<2x4x16xf32> into tensor<8x16xf32>
  iree_codegen.store_to_buffer %collapse, %result : tensor<8x16xf32> into memref<8x16xf32>
  return
}
// DISPATCH-SCOPE-LABEL: @fold_collapse_shape_op
//  DISPATCH-SCOPE-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  DISPATCH-SCOPE-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   DISPATCH-SCOPE-DAG:   %[[TRUE:.+]] = arith.constant true
//       DISPATCH-SCOPE:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<8x16xf32>
//       DISPATCH-SCOPE:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  DISPATCH-SCOPE-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  DISPATCH-SCOPE-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index):
//       DISPATCH-SCOPE:     %[[LINEARIZE:.+]] = affine.linearize_index
//  DISPATCH-SCOPE-SAME:       [%[[IDX0]], %[[IDX1]]] by (2, 4)
//       DISPATCH-SCOPE:     iree_linalg_ext.yield %[[LINEARIZE]], %[[IDX2]], %[[TRUE]]
//       DISPATCH-SCOPE:   } : tensor<2x4x16xf32> into tensor<8x16xf32> -> tensor<8x16xf32>
//       DISPATCH-SCOPE:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<8x16xf32> into memref<8x16xf32>

// -----

func.func @fold_expand_shape_op(%source : tensor<8x16xf32>, %result : memref<2x4x16xf32>) {
  %expand = tensor.expand_shape %source [[0, 1], [2]] output_shape [2, 4, 16] : tensor<8x16xf32> into tensor<2x4x16xf32>
  iree_codegen.store_to_buffer %expand, %result : tensor<2x4x16xf32> into memref<2x4x16xf32>
  return
}
// DISPATCH-SCOPE-LABEL: @fold_expand_shape_op
//  DISPATCH-SCOPE-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  DISPATCH-SCOPE-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   DISPATCH-SCOPE-DAG:   %[[TRUE:.+]] = arith.constant true
//       DISPATCH-SCOPE:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<2x4x16xf32>
//       DISPATCH-SCOPE:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  DISPATCH-SCOPE-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  DISPATCH-SCOPE-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//       DISPATCH-SCOPE:     %[[DELINEARIZE:.+]]:2 = affine.delinearize_index %[[IDX0]] into (2, 4)
//       DISPATCH-SCOPE:     iree_linalg_ext.yield %[[DELINEARIZE]]#0, %[[DELINEARIZE]]#1, %[[IDX1]], %[[TRUE]]
//       DISPATCH-SCOPE:   } : tensor<8x16xf32> into tensor<2x4x16xf32> -> tensor<2x4x16xf32>
//       DISPATCH-SCOPE:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<2x4x16xf32> into memref<2x4x16xf32>

// -----

func.func @fold_transpose_op(%source : tensor<2x4x16xf32>, %result : memref<4x16x2xf32>) {
  %init = tensor.empty() : tensor<4x16x2xf32>
  %transposed = linalg.transpose ins(%source : tensor<2x4x16xf32>) outs(%init : tensor<4x16x2xf32>) permutation = [1, 2, 0]
  iree_codegen.store_to_buffer %transposed, %result : tensor<4x16x2xf32> into memref<4x16x2xf32>
  return
}
// DISPATCH-SCOPE-LABEL: @fold_transpose_op
//  DISPATCH-SCOPE-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  DISPATCH-SCOPE-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   DISPATCH-SCOPE-DAG:   %[[TRUE:.+]] = arith.constant true
//       DISPATCH-SCOPE:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<4x16x2xf32>
//       DISPATCH-SCOPE:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  DISPATCH-SCOPE-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  DISPATCH-SCOPE-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index):
//       DISPATCH-SCOPE:     iree_linalg_ext.yield %[[IDX1]], %[[IDX2]], %[[IDX0]], %[[TRUE]]
//       DISPATCH-SCOPE:   } : tensor<2x4x16xf32> into tensor<4x16x2xf32> -> tensor<4x16x2xf32>
//       DISPATCH-SCOPE:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<4x16x2xf32> into memref<4x16x2xf32>

// -----

func.func @fold_extract_slice_op(%source : tensor<64xf32>, %result : memref<63xf32>) {
  %slice = tensor.extract_slice %source[0] [63] [1] : tensor<64xf32> to tensor<63xf32>
  iree_codegen.store_to_buffer %slice, %result : tensor<63xf32> into memref<63xf32>
  return
}
// DISPATCH-SCOPE-LABEL: @fold_extract_slice_op
//  DISPATCH-SCOPE-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  DISPATCH-SCOPE-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   DISPATCH-SCOPE-DAG:   %[[C63:.+]] = arith.constant 63 : index
//       DISPATCH-SCOPE:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<63xf32>
//       DISPATCH-SCOPE:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  DISPATCH-SCOPE-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  DISPATCH-SCOPE-NEXT:   ^bb0(%[[IDX0:.+]]: index):
//       DISPATCH-SCOPE:     %[[MASK:.+]] = arith.cmpi ult, %[[IDX0]], %[[C63]]
//       DISPATCH-SCOPE:     iree_linalg_ext.yield %[[IDX0]], %[[MASK]]
//       DISPATCH-SCOPE:   } : tensor<64xf32> into tensor<63xf32> -> tensor<63xf32>
//       DISPATCH-SCOPE:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<63xf32> into memref<63xf32>

// -----

func.func @no_fold_offset_extract_slice_op(%source : tensor<64xf32>, %result : memref<4xf32>) {
  %slice = tensor.extract_slice %source[42] [4] [1] : tensor<64xf32> to tensor<4xf32>
  iree_codegen.store_to_buffer %slice, %result : tensor<4xf32> into memref<4xf32>
  return
}
// DISPATCH-SCOPE-LABEL: @no_fold_offset_extract_slice_op
//       DISPATCH-SCOPE:   tensor.extract_slice
//   DISPATCH-SCOPE-NOT:   iree_linalg_ext.map_scatter

// -----

func.func @no_fold_strided_extract_slice_op(%source : tensor<64xf32>, %result : memref<16xf32>) {
  %slice = tensor.extract_slice %source[0] [16] [4] : tensor<64xf32> to tensor<16xf32>
  iree_codegen.store_to_buffer %slice, %result : tensor<16xf32> into memref<16xf32>
  return
}
// DISPATCH-SCOPE-LABEL: @no_fold_strided_extract_slice_op
//       DISPATCH-SCOPE:   tensor.extract_slice
//   DISPATCH-SCOPE-NOT:   iree_linalg_ext.map_scatter

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
//       DISPATCH-SCOPE: #[[$MAP:.+]] = affine_map<(d0) -> (256, d0 + 64)>
// DISPATCH-SCOPE-LABEL: @fold_pad_op
//  DISPATCH-SCOPE-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  DISPATCH-SCOPE-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   DISPATCH-SCOPE-DAG:   %[[PAD_VAL:.+]] = arith.constant 0.000000e+00 : f32
//   DISPATCH-SCOPE-DAG:   %[[TRUE:.+]] = arith.constant true
//   DISPATCH-SCOPE-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   DISPATCH-SCOPE-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   DISPATCH-SCOPE-DAG:   %[[C252:.+]] = arith.constant 252 : index
//       DISPATCH-SCOPE:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<256xf32>
//       DISPATCH-SCOPE:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  DISPATCH-SCOPE-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  DISPATCH-SCOPE-NEXT:   ^bb0(%[[IDX0:.+]]: index):
//       DISPATCH-SCOPE:     iree_linalg_ext.yield %[[IDX0]], %[[TRUE]]
//       DISPATCH-SCOPE:   } : tensor<250xf32> into tensor<256xf32> -> tensor<256xf32>
//       DISPATCH-SCOPE:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<256xf32> into memref<256xf32>

//       DISPATCH-SCOPE:   scf.forall (%[[WG_IV:.+]]) = (0) to (256) step (64) {
//       DISPATCH-SCOPE:     %[[WG_TILE_UB:.+]] = affine.min #[[$MAP]](%[[WG_IV]])
//       DISPATCH-SCOPE:     scf.for %[[IDX:.+]] = %[[WG_IV]] to %[[WG_TILE_UB]] step %[[C1]] {
//   DISPATCH-SCOPE-DAG:       %[[IS_LOW_PAD:.+]] = arith.cmpi ult, %[[IDX]], %[[C2]]
//   DISPATCH-SCOPE-DAG:       %[[IS_HIGH_PAD:.+]] = arith.cmpi uge, %[[IDX]], %[[C252]]
//   DISPATCH-SCOPE-DAG:       %[[IS_PAD:.+]] = arith.ori %[[IS_LOW_PAD]], %[[IS_HIGH_PAD]] : i1
//       DISPATCH-SCOPE:       scf.if %[[IS_PAD]] {
//  DISPATCH-SCOPE-NEXT:         memref.store %[[PAD_VAL]], %[[RESULT]][%[[IDX]]] : memref<256xf32>
//  DISPATCH-SCOPE-NEXT:       }
//  DISPATCH-SCOPE:          }
//  DISPATCH-SCOPE:        } {mapping = [#iree_codegen.workgroup_mapping<x>]}

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
//       DISPATCH-SCOPE: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 * 128)>
// DISPATCH-SCOPE-LABEL: @fold_unpack_op
//  DISPATCH-SCOPE-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  DISPATCH-SCOPE-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   DISPATCH-SCOPE-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   DISPATCH-SCOPE-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   DISPATCH-SCOPE-DAG:   %[[RES_D0:.+]] = memref.dim %[[RESULT]], %[[C0]] : memref<?x?xf32>
//   DISPATCH-SCOPE-DAG:   %[[RES_D1:.+]] = memref.dim %[[RESULT]], %[[C1]] : memref<?x?xf32>
//   DISPATCH-SCOPE-DAG:   %[[SRC_D0:.+]] = tensor.dim %[[SOURCE]], %[[C0]] : tensor<?x?x128x128xf32>
//   DISPATCH-SCOPE-DAG:   %[[SRC_D1:.+]] = tensor.dim %[[SOURCE]], %[[C1]] : tensor<?x?x128x128xf32>
//   DISPATCH-SCOPE-DAG:   %[[COLLAPSE_SIZE0:.+]] = affine.apply #[[$MAP]]()[%[[SRC_D0]]]
//   DISPATCH-SCOPE-DAG:   %[[COLLAPSE_SIZE1:.+]] = affine.apply #[[$MAP]]()[%[[SRC_D1]]]
//       DISPATCH-SCOPE:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty(%[[RES_D0]], %[[RES_D1]]) : tensor<?x?xf32>
//       DISPATCH-SCOPE:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  DISPATCH-SCOPE-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  DISPATCH-SCOPE-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index, %[[IDX3:.+]]: index):
//       DISPATCH-SCOPE:     %[[LINEARIZE:.+]] = affine.linearize_index
//  DISPATCH-SCOPE-SAME:       [%[[IDX0]], %[[IDX2]], %[[IDX1]], %[[IDX3]]]
//  DISPATCH-SCOPE-SAME:       by (%[[SRC_D0]], 128, %[[SRC_D1]], 128)
//       DISPATCH-SCOPE:     %[[DELINEARIZE:.+]]:2 = affine.delinearize_index %[[LINEARIZE]]
//  DISPATCH-SCOPE-SAME:       into (%[[COLLAPSE_SIZE0]], %[[COLLAPSE_SIZE1]])
//       DISPATCH-SCOPE:     %[[BOUND0:.+]] = arith.cmpi ult, %[[DELINEARIZE]]#0, %[[RES_D0]]
//       DISPATCH-SCOPE:     %[[BOUND1:.+]] = arith.cmpi ult, %[[DELINEARIZE]]#1, %[[RES_D1]]
//       DISPATCH-SCOPE:     %[[MASK:.+]] = arith.andi %[[BOUND0]], %[[BOUND1]] : i1
//       DISPATCH-SCOPE:     iree_linalg_ext.yield %[[DELINEARIZE]]#0, %[[DELINEARIZE]]#1, %[[MASK]]
//       DISPATCH-SCOPE:   } : tensor<?x?x128x128xf32> into tensor<?x?xf32> -> tensor<?x?xf32>
//       DISPATCH-SCOPE:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<?x?xf32> into memref<?x?xf32>

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
//       DISPATCH-SCOPE: #[[$MAP:.+]] = affine_map<(d0) -> (256, d0 + 1)>
//       DISPATCH-SCOPE: #[[$MAP1:.+]] = affine_map<(d0) -> (256, d0 + 64)>
// DISPATCH-SCOPE-LABEL: @fold_pack_op
//  DISPATCH-SCOPE-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  DISPATCH-SCOPE-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   DISPATCH-SCOPE-DAG:   %[[TRUE:.+]] = arith.constant true
//   DISPATCH-SCOPE-DAG:   %[[PAD_VAL:.+]] = arith.constant 0.000000e+00 : f32
//   DISPATCH-SCOPE-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   DISPATCH-SCOPE-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   DISPATCH-SCOPE-DAG:   %[[C250:.+]] = arith.constant 250 : index
//       DISPATCH-SCOPE:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<2x2x128x128xf32>
//       DISPATCH-SCOPE:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  DISPATCH-SCOPE-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  DISPATCH-SCOPE-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//       DISPATCH-SCOPE:     %[[DELINEARIZE0:.+]]:2 = affine.delinearize_index %[[IDX0]]
//  DISPATCH-SCOPE-SAME:       into (2, 128)
//       DISPATCH-SCOPE:     %[[DELINEARIZE1:.+]]:2 = affine.delinearize_index %[[IDX1]]
//  DISPATCH-SCOPE-SAME:       into (2, 128)
//       DISPATCH-SCOPE:     iree_linalg_ext.yield %[[DELINEARIZE0]]#0, %[[DELINEARIZE1]]#0, %[[DELINEARIZE0]]#1, %[[DELINEARIZE1]]#1, %[[TRUE]]
//       DISPATCH-SCOPE:   } : tensor<250x250xf32> into tensor<2x2x128x128xf32> -> tensor<2x2x128x128xf32>
//       DISPATCH-SCOPE:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<2x2x128x128xf32> into memref<2x2x128x128xf32>

//       DISPATCH-SCOPE:   scf.forall (%[[WG_IV0:.+]], %[[WG_IV1:.+]]) = (0, 0) to (256, 256) step (1, 64) {
//   DISPATCH-SCOPE-DAG:     %[[WG_TILE_UB0:.+]] = affine.min #[[$MAP]](%[[WG_IV0]])
//   DISPATCH-SCOPE-DAG:     %[[WG_TILE_UB1:.+]] = affine.min #[[$MAP1]](%[[WG_IV1]])
//       DISPATCH-SCOPE:     scf.for %[[IDX0:.+]] = %[[WG_IV0]] to %[[WG_TILE_UB0]] step %[[C1]] {
//       DISPATCH-SCOPE:       scf.for %[[IDX1:.+]] = %[[WG_IV1]] to %[[WG_TILE_UB1]] step %[[C1]] {
//   DISPATCH-SCOPE-DAG:         %[[EXPANDED_IDX0:.+]]:2 = affine.delinearize_index %[[IDX0]] into (2, 128)
//   DISPATCH-SCOPE-DAG:         %[[EXPANDED_IDX1:.+]]:2 = affine.delinearize_index %[[IDX1]] into (2, 128)
//   DISPATCH-SCOPE-DAG:         %[[IDX0_IS_LOW_PAD:.+]] = arith.cmpi ult, %[[IDX0]], %[[C0]]
//   DISPATCH-SCOPE-DAG:         %[[IDX0_IS_HIGH_PAD:.+]] = arith.cmpi uge, %[[IDX0]], %[[C250]]
//   DISPATCH-SCOPE-DAG:         %[[IDX0_IS_PAD:.+]] = arith.ori %[[IDX0_IS_LOW_PAD]], %[[IDX0_IS_HIGH_PAD]] : i1
//   DISPATCH-SCOPE-DAG:         %[[IDX1_IS_LOW_PAD:.+]] = arith.cmpi ult, %[[IDX1]], %[[C0]]
//   DISPATCH-SCOPE-DAG:         %[[IDX1_IS_HIGH_PAD:.+]] = arith.cmpi uge, %[[IDX1]], %[[C250]]
//   DISPATCH-SCOPE-DAG:         %[[IDX1_IS_PAD:.+]] = arith.ori %[[IDX1_IS_LOW_PAD]], %[[IDX1_IS_HIGH_PAD]] : i1
//   DISPATCH-SCOPE-DAG:         %[[IS_PAD:.+]] = arith.ori %[[IDX0_IS_PAD]], %[[IDX1_IS_PAD]] : i1
//       DISPATCH-SCOPE:         scf.if %[[IS_PAD]] {
//  DISPATCH-SCOPE-NEXT:           memref.store %[[PAD_VAL]], %[[RESULT]]
//  DISPATCH-SCOPE-SAME:             [%[[EXPANDED_IDX0]]#0, %[[EXPANDED_IDX1]]#0, %[[EXPANDED_IDX0]]#1, %[[EXPANDED_IDX1]]#1]
//  DISPATCH-SCOPE-SAME:             : memref<2x2x128x128xf32>
//  DISPATCH-SCOPE-NEXT:         }
//  DISPATCH-SCOPE:            }
//  DISPATCH-SCOPE:          }
//  DISPATCH-SCOPE:        } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

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
// DISPATCH-SCOPE-LABEL: @propagate_relayout_ops
//  DISPATCH-SCOPE-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//       DISPATCH-SCOPE:   %[[INIT:.+]] = tensor.empty{{.*}} : tensor<?x?x128x128xf16>
//       DISPATCH-SCOPE:   %[[COMPUTE_OP:.+]] = linalg.generic
//  DISPATCH-SCOPE-SAME:     ins(%[[SOURCE]] : tensor<?x?x128x128xf32>)
//  DISPATCH-SCOPE-SAME:     outs(%[[INIT]] : tensor<?x?x128x128xf16>)
//       DISPATCH-SCOPE:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter %[[COMPUTE_OP]]
//       DISPATCH-SCOPE:   iree_codegen.store_to_buffer %[[MAP_SCATTER]]

// -----

func.func @insert_in_workgroup_forall(%2 : tensor<32xbf16>, %3 : tensor<32xbf16>, %9 : tensor<10xbf16>) -> (tensor<32xbf16>, tensor<32xbf16>) {
  %6:2 = scf.forall (%arg0) = (0) to (32) step (8) shared_outs(%arg2 = %2, %arg3 = %3) -> (tensor<32xbf16>, tensor<32xbf16>) {
    %extract = tensor.extract_slice %9 [0] [8] [1] : tensor<10xbf16> to tensor<8xbf16>
    %extract_0 = tensor.extract_slice %9 [0] [7] [1] : tensor<10xbf16> to tensor<7xbf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %extract into %arg2[%arg0] [8] [1] : tensor<8xbf16> into tensor<32xbf16>
      tensor.parallel_insert_slice %extract_0 into %arg3[%arg0] [7] [1] : tensor<7xbf16> into tensor<32xbf16>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  return %6#0, %6#1 : tensor<32xbf16>, tensor<32xbf16>
}

//   WORKGROUP-SCOPE-LABEL: @insert_in_workgroup_forall
//         WORKGROUP-SCOPE:   scf.forall
// WORKGROUP-SCOPE-COUNT-2:     iree_linalg_ext.map_scatter
//         WORKGROUP-SCOPE:   } {mapping = [#iree_codegen.workgroup_mapping<x>]}

// -----

func.func @no_insert_reshape_only(%2 : tensor<196608x35xbf16>, %9 : tensor<8x16x1x16xbf16>) -> tensor<196608x35xbf16> {
  %6 = scf.forall (%arg0, %arg1) = (0, 0) to (196608, 35) step (128, 16) shared_outs(%arg2 = %2) -> tensor<196608x35xbf16> {
    %collapsed = tensor.collapse_shape %9 [[0, 1], [2, 3]] : tensor<8x16x1x16xbf16> into tensor<128x16xbf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %collapsed into %arg2[%arg0, %arg1] [128, 16] [1, 1] : tensor<128x16xbf16> into tensor<196608x35xbf16>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  return %6  : tensor<196608x35xbf16>
}

// WORKGROUP-SCOPE-LABEL: @no_insert_reshape_only
//   WORKGROUP-SCOPE-NOT:   iree_linalg_ext.map_scatter

// -----

func.func @no_insert_in_non_workgroup_forall(%2 : tensor<32xbf16>, %9 : tensor<10xbf16>) -> tensor<32xbf16>{
  %6 = scf.forall (%arg0) = (0) to (32) step (8) shared_outs(%arg2 = %2) -> (tensor<32xbf16>) {
    %extract = tensor.extract_slice %9 [0] [8] [1] : tensor<10xbf16> to tensor<8xbf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %extract into %arg2[%arg0] [8] [1] : tensor<8xbf16> into tensor<32xbf16>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return %6 : tensor<32xbf16>
}
// WORKGROUP-SCOPE-LABEL: @no_insert_in_non_workgroup_forall
//   WORKGROUP-SCOPE-NOT:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter

// -----

func.func @workgroup_and_dispatch_scope(%arg0 : tensor<32xbf16>, %arg1 : tensor<10xbf16>, %arg2 : memref<20xbf16>) {
  %0 = tensor.empty() : tensor<32xbf16>
  %1 = scf.forall (%arg3) = (0) to (32) step (8) shared_outs(%arg4 = %0) -> tensor<32xbf16> {
    %extract = tensor.extract_slice %arg1 [0] [8] [1] : tensor<10xbf16> to tensor<8xbf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %extract into %arg4[%arg3] [8] [1] : tensor<8xbf16> into tensor<32xbf16>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  %barrier = util.optimization_barrier %1 : tensor<32xbf16>
  %extract = tensor.extract_slice %barrier [0] [20] [1] : tensor<32xbf16> to tensor<20xbf16>
  iree_codegen.store_to_buffer %extract, %arg2 : tensor<20xbf16> into memref<20xbf16>
  return
}

// DISPATCH-SCOPE-LABEL: @workgroup_and_dispatch_scope
//       DISPATCH-SCOPE:   scf.forall
//       DISPATCH-SCOPE:     tensor.extract_slice
//       DISPATCH-SCOPE:   } {mapping = [#iree_codegen.workgroup_mapping<x>]}
//       DISPATCH-SCOPE:   iree_linalg_ext.map_scatter

// WORKGROUP-SCOPE-LABEL: @workgroup_and_dispatch_scope
//       WORKGROUP-SCOPE:   scf.forall
//       WORKGROUP-SCOPE:     iree_linalg_ext.map_scatter
//       WORKGROUP-SCOPE:   } {mapping = [#iree_codegen.workgroup_mapping<x>]}
//   WORKGROUP-SCOPE-NOT:   iree_linalg_ext.map_scatter
//       WORKGROUP-SCOPE:   tensor.extract_slice

// -----

// Tests that the consumer fusion pattern stops after fusion fails, and the
// consumer ops are not added back to the worklist (more context in #21622).

#map = affine_map<(d0) -> (d0 * 8)>
func.func @consumer_unfusable_due_to_init(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
  %c0 = arith.constant 0 : index
  %1 = scf.forall (%arg2) in (64) shared_outs(%arg3 = %arg1) -> (tensor<?xi32>) {
    %2 = affine.apply #map(%arg2)
    %extracted_slice = tensor.extract_slice %arg0[%2] [8] [1] : tensor<?xi32> to tensor<8xi32>
    %extracted_slice0 = tensor.extract_slice %arg3[%2] [8] [1] : tensor<?xi32> to tensor<8xi32>
    %copied = linalg.copy ins(%extracted_slice : tensor<8xi32>) outs(%extracted_slice0 : tensor<8xi32>) -> tensor<8xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %copied into %arg3[%2] [8] [1] : tensor<8xi32> into tensor<?xi32>
    }
  }
  %barrier = util.optimization_barrier %1 : tensor<?xi32>
  %dim = tensor.dim %barrier, %c0 : tensor<?xi32>
  %empty = tensor.empty(%dim) : tensor<?xi32>
  %consumer = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
      ins(%1 : tensor<?xi32>) outs(%empty : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %add = arith.addi %in, %in : i32
    linalg.yield %add : i32
  } -> tensor<?xi32>
  return %consumer : tensor<?xi32>
}

// DISPATCH-SCOPE-LABEL: func @consumer_unfusable_due_to_init
//       DISPATCH-SCOPE:   scf.forall
//       DISPATCH-SCOPE:     linalg.copy
//       DISPATCH-SCOPE:     scf.forall.in_parallel
//       DISPATCH-SCOPE:       tensor.parallel_insert_slice
//       DISPATCH-SCOPE:   linalg.generic
