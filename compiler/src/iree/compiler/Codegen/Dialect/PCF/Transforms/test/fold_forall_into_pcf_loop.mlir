// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-test-fold-forall-into-pcf-loop)" --mlir-print-local-scope --split-input-file | FileCheck %s

// Test folding scf.forall containing pcf.loop into a single pcf.generic.
// Forall has 2D iteration space (4, 8) with loop count 4.
// The write_slice writes at [loop_id, 0] with size [1, 4] into the 4x4 tile.
// The parallel_insert_slice inserts at [id0, id1] with size [4, 4].
// Composed write should be at [loop_id + id0, id1] with size [1, 4] strides [1, 1].

func.func @fold_forall_into_pcf_loop(%init: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %c4 = arith.constant 4 : index
  %0 = scf.forall (%id0, %id1) in (4, 8) shared_outs(%iter = %init) -> (tensor<16x32xf32>) {
    %tile_init = tensor.extract_slice %iter[%id0, %id1] [4, 4] [1, 1]
        : tensor<16x32xf32> to tensor<4x4xf32>
    %loop_result = pcf.loop scope(#pcf.sequential) count(%c4)
        execute(%ref = %tile_init)[%loop_id: index]
            : (!pcf.sref<4x4xf32, sync(#pcf.sequential)>)
           -> (tensor<4x4xf32>) {
      %slice = tensor.extract_slice %init[%id0, %loop_id] [1, 4] [1, 1]
          : tensor<16x32xf32> to tensor<1x4xf32>
      pcf.write_slice %slice into %ref[%loop_id, 0] [1, 4] [1, 1]
          : tensor<1x4xf32> into !pcf.sref<4x4xf32, sync(#pcf.sequential)>
      pcf.return
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %loop_result into %iter[%id0, %id1] [4, 4] [1, 1]
          : tensor<4x4xf32> into tensor<16x32xf32>
    }
  } {mapping = [#iree_codegen.local_mapping<0>, #iree_codegen.local_mapping<1>]}
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: @fold_forall_into_pcf_loop
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<16x32xf32>

//       CHECK:   %[[GENERIC:.+]] = pcf.generic
//       CHECK:     scope(#pcf.sequential)
//       CHECK:     execute(%[[REF:[A-Za-z0-9_]+]] = %[[INIT]])[%[[GEN_ID:[A-Za-z0-9_]+]]: index, %[[GEN_COUNT:[A-Za-z0-9_]+]]: index]
//       CHECK:          : (!pcf.sref<16x32xf32, sync(#pcf.sequential)>)
//       CHECK:         -> (tensor<16x32xf32>) {

// Delinearize generic id into (forall_dim0, forall_dim1, pcf_loop_id) with
// full basis [4, 8, loop_count].
//       CHECK:     %[[DELIN:.+]]:3 = affine.delinearize_index %[[GEN_ID]] into
//       CHECK-SAME:  : index, index, index

// Linearize forall dim indices back to forall_linear_id (cancels with delinearize).
//       CHECK:     %[[FORALL_LIN:.+]] = affine.linearize_index disjoint [%[[DELIN]]#0, %[[DELIN]]#1] by (4, 8)

// Outer scf.forall starts at forall_linear_id, upper bound = 4*8 = 32.
//       CHECK:     scf.forall (%[[OUTER_IV:.+]]) = (%[[FORALL_LIN]]) to
//       CHECK-SAME:  {

// Delinearize outer IV into 2D forall space (4, 8).
//       CHECK:       %[[FORALL_DELIN:.+]]:2 = affine.delinearize_index %[[OUTER_IV]] into
//       CHECK-SAME:    : index, index

// Inner scf.forall starts at pcf_loop_id, upper bound = loop count.
//       CHECK:       scf.forall (%[[INNER_IV:.+]]) = (%[[DELIN]]#2) to
//       CHECK-SAME:    {

// Composed write: sizes [1, 4] and strides [1, 1] from write_slice.
// Offset dim 0 = insertOff(id0) + writeOff(loop_id) * insertStride(1).
// Offset dim 1 = insertOff(id1) + writeOff(0) * insertStride(1) = id1.
//       CHECK:         %[[COMPOSED_OFF_0:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%[[FORALL_DELIN]]#0, %[[INNER_IV]]]
//       CHECK:         pcf.write_slice %{{.+}} into %[[REF]][%[[COMPOSED_OFF_0]], %[[FORALL_DELIN]]#1]
//  CHECK-SAME:           [1, 4] [1, 1]
//  CHECK-SAME:           into !pcf.sref<16x32xf32, sync(#pcf.sequential)>
//       CHECK:     pcf.return
//       CHECK:   return %[[GENERIC]]

// -----

// Test with multiple results from pcf.loop.
// Both results get composed writes targeting different ref args.

func.func @fold_forall_multiple_results(%init0: tensor<16xf32>, %init1: tensor<16xf32>)
    -> (tensor<16xf32>, tensor<16xf32>) {
  %c2 = arith.constant 2 : index
  %0:2 = scf.forall (%id) in (4) shared_outs(%iter0 = %init0, %iter1 = %init1)
      -> (tensor<16xf32>, tensor<16xf32>) {
    %tile_init0 = tensor.extract_slice %iter0[%id] [4] [1]
        : tensor<16xf32> to tensor<4xf32>
    %tile_init1 = tensor.extract_slice %iter1[%id] [4] [1]
        : tensor<16xf32> to tensor<4xf32>
    %loop_result:2 = pcf.loop scope(#pcf.sequential) count(%c2)
        execute(%ref0 = %tile_init0, %ref1 = %tile_init1)[%loop_id: index]
            : (!pcf.sref<4xf32, sync(#pcf.sequential)>,
               !pcf.sref<4xf32, sync(#pcf.sequential)>)
           -> (tensor<4xf32>, tensor<4xf32>) {
      %slice0 = tensor.extract_slice %init0[%loop_id] [2] [1]
          : tensor<16xf32> to tensor<2xf32>
      %slice1 = tensor.extract_slice %init1[%loop_id] [2] [1]
          : tensor<16xf32> to tensor<2xf32>
      pcf.write_slice %slice0 into %ref0[%loop_id] [2] [1]
          : tensor<2xf32> into !pcf.sref<4xf32, sync(#pcf.sequential)>
      pcf.write_slice %slice1 into %ref1[%loop_id] [2] [1]
          : tensor<2xf32> into !pcf.sref<4xf32, sync(#pcf.sequential)>
      pcf.return
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %loop_result#0 into %iter0[%id] [4] [1]
          : tensor<4xf32> into tensor<16xf32>
      tensor.parallel_insert_slice %loop_result#1 into %iter1[%id] [4] [1]
          : tensor<4xf32> into tensor<16xf32>
    }
  } {mapping = [#iree_codegen.local_mapping<0>]}
  return %0#0, %0#1 : tensor<16xf32>, tensor<16xf32>
}

// CHECK-LABEL: @fold_forall_multiple_results
//  CHECK-SAME:   %[[INIT0:[A-Za-z0-9_]+]]: tensor<16xf32>
//  CHECK-SAME:   %[[INIT1:[A-Za-z0-9_]+]]: tensor<16xf32>

//       CHECK:   %[[GENERIC:.+]]:2 = pcf.generic
//       CHECK:     scope(#pcf.sequential)
//       CHECK:     execute(%[[REF0:[A-Za-z0-9_]+]] = %[[INIT0]], %[[REF1:[A-Za-z0-9_]+]] = %[[INIT1]])[%[[GEN_ID:[A-Za-z0-9_]+]]: index, %{{.*}}: index]
//       CHECK:          : (!pcf.sref<16xf32, sync(#pcf.sequential)>, !pcf.sref<16xf32, sync(#pcf.sequential)>)
//       CHECK:         -> (tensor<16xf32>, tensor<16xf32>) {

// Delinearize generic id into (forall_dim0, pcf_loop_id).
//       CHECK:     %[[DELIN:.+]]:2 = affine.delinearize_index %[[GEN_ID]] into
//       CHECK-SAME:  : index, index

// Linearize forall dim index (1D, so linearize is trivial).
//       CHECK:     %[[FORALL_LIN:.+]] = affine.linearize_index disjoint [%[[DELIN]]#0] by (4)

// Outer scf.forall from forall_linear_id.
//       CHECK:     scf.forall (%{{.+}}) = (%[[FORALL_LIN]])

// Inner scf.forall from pcf_loop_id.
//       CHECK:       scf.forall (%{{.+}}) = (%[[DELIN]]#1)

// Composed writes: write offset[loop_id] + insert offset[id].
// Both writes have size 2 and stride 1, targeting different ref args.
//       CHECK:         %[[COMPOSED_OFF_0:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%{{.+}}, %{{.+}}]
//       CHECK:         pcf.write_slice %{{.+}} into %[[REF0]][%[[COMPOSED_OFF_0]]] [2] [1]
//  CHECK-SAME:           into !pcf.sref<16xf32, sync(#pcf.sequential)>
//       CHECK:         %[[COMPOSED_OFF_1:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%{{.+}}, %{{.+}}]
//       CHECK:         pcf.write_slice %{{.+}} into %[[REF1]][%[[COMPOSED_OFF_1]]] [2] [1]
//  CHECK-SAME:           into !pcf.sref<16xf32, sync(#pcf.sequential)>
//       CHECK:     pcf.return
//       CHECK:   return %[[GENERIC]]#0, %[[GENERIC]]#1

// -----

// Test write_slice + parallel_insert_slice composition with non-unit write
// stride. The write has stride 2, the insert has stride 1.
// Composed stride = 2 * 1 = 2. Write size = 2, stays 2.

func.func @fold_compose_strides(%init: tensor<64xf32>) -> tensor<64xf32> {
  %c3 = arith.constant 3 : index
  %0 = scf.forall (%id) in (4) shared_outs(%iter = %init) -> (tensor<64xf32>) {
    %tile_init = tensor.extract_slice %iter[%id] [16] [1]
        : tensor<64xf32> to tensor<16xf32>
    %loop_result = pcf.loop scope(#pcf.sequential) count(%c3)
        execute(%ref = %tile_init)[%loop_id: index]
            : (!pcf.sref<16xf32, sync(#pcf.sequential)>)
           -> (tensor<16xf32>) {
      %val = tensor.extract_slice %init[0] [2] [1]
          : tensor<64xf32> to tensor<2xf32>
      pcf.write_slice %val into %ref[1] [2] [2]
          : tensor<2xf32> into !pcf.sref<16xf32, sync(#pcf.sequential)>
      pcf.return
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %loop_result into %iter[%id] [16] [1]
          : tensor<16xf32> into tensor<64xf32>
    }
  } {mapping = [#iree_codegen.local_mapping<0>]}
  return %0 : tensor<64xf32>
}

// Composed offset = insert(id) + write(1) * insert_stride(1) = id + 1.
// Size = 2 (from write). Stride = 2 * 1 = 2.
// CHECK-LABEL: @fold_compose_strides
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<64xf32>
//       CHECK:   %[[GENERIC:.+]] = pcf.generic
//       CHECK:     execute(%[[REF:[A-Za-z0-9_]+]] = %[[INIT]])[%[[GEN_ID:[A-Za-z0-9_]+]]: index
//       CHECK:          : (!pcf.sref<64xf32, sync(#pcf.sequential)>)
//       CHECK:     %[[DELIN:.+]]:2 = affine.delinearize_index %[[GEN_ID]] into
//       CHECK:     %[[FORALL_LIN:.+]] = affine.linearize_index disjoint
//       CHECK:     %[[OUTER_STEP:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 3)>()[%{{.+}}]
//       CHECK:     scf.forall (%[[OUTER_IV:.+]]) = (%[[FORALL_LIN]]) to (%{{.+}}) step (%[[OUTER_STEP]])
//       CHECK:       %[[INNER_STEP:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 3)>()[%{{.+}}]
//       CHECK:       scf.forall (%[[INNER_IV:.+]]) = (%[[DELIN]]#1) to (%{{.+}}) step (%[[INNER_STEP]])
// Composed: size 2, stride 2 (write_stride=2 * insert_stride=1).
//       CHECK:         %[[COMPOSED_OFF:.+]] = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%[[OUTER_IV]]]
//       CHECK:         pcf.write_slice %{{.+}} into %[[REF]][%[[COMPOSED_OFF]]] [2] [2]
//  CHECK-SAME:           into !pcf.sref<64xf32, sync(#pcf.sequential)>
//       CHECK:     pcf.return

// -----

// Test folding when the pcf.loop count is computed in the forall body and
// needs to be hoisted before rewriting.
func.func @fold_hoists_loop_count(%init: tensor<8xf32>, %one: index)
    -> tensor<8xf32> {
  %0 = scf.forall (%id) in (4) shared_outs(%iter = %init) -> (tensor<8xf32>) {
    %tile_init = tensor.extract_slice %iter[%id] [2] [1]
        : tensor<8xf32> to tensor<2xf32>
    %count = arith.addi %one, %one : index
    %loop_result = pcf.loop scope(#pcf.sequential) count(%count)
        execute(%ref = %tile_init)[%loop_id: index]
            : (!pcf.sref<2xf32, sync(#pcf.sequential)>)
           -> (tensor<2xf32>) {
      %slice = tensor.extract_slice %init[%loop_id] [1] [1]
          : tensor<8xf32> to tensor<1xf32>
      pcf.write_slice %slice into %ref[%loop_id] [1] [1]
          : tensor<1xf32> into !pcf.sref<2xf32, sync(#pcf.sequential)>
      pcf.return
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %loop_result into %iter[%id] [2] [1]
          : tensor<2xf32> into tensor<8xf32>
    }
  } {mapping = [#iree_codegen.local_mapping<0>]}
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @fold_hoists_loop_count
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<8xf32>, %[[ONE:[A-Za-z0-9_]+]]: index
//       CHECK:   %[[COUNT:[A-Za-z0-9_]+]] = arith.addi %[[ONE]], %[[ONE]] : index
//       CHECK:   %[[GENERIC:.+]] = pcf.generic
//       CHECK:     execute(%{{.+}} = %[[INIT]])[%[[GEN_ID:[A-Za-z0-9_]+]]: index, %[[GEN_COUNT:[A-Za-z0-9_]+]]: index]
//       CHECK:     %[[DELIN:.+]]:2 = affine.delinearize_index %[[GEN_ID]] into (4, %[[COUNT]])
//       CHECK:     %[[OUTER_STEP:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 ceildiv s1)>()[%[[GEN_COUNT]], %[[COUNT]]]
//       CHECK:       scf.forall (%{{.+}}) = (%{{.+}}) to (%{{.+}}) step (%[[OUTER_STEP]])
//       CHECK:       %[[INNER_STEP:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 ceildiv s1)>()[%[[GEN_COUNT]], %[[COUNT]]]
//       CHECK:       scf.forall (%{{.+}}) = (%[[DELIN]]#1) to (%[[COUNT]]) step (%[[INNER_STEP]])
