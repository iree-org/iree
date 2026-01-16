// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-pcf-test-convert-forall-to-generic-nest{num-sequential-scopes=1}))" --split-input-file | FileCheck %s --check-prefix=CHECK-1SCOPE
// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-pcf-test-convert-forall-to-generic-nest{num-sequential-scopes=2}))" --split-input-file | FileCheck %s --check-prefix=CHECK-2SCOPE

// Single scope: one pcf.generic wrapping scf.forall.
// Two scopes: nested pcf.generic ops with linearized ids.

// CHECK-1SCOPE-LABEL: func.func @test_1d_single_scope
// CHECK-1SCOPE-SAME:    %[[INIT:.+]]: tensor<64xf32>
// CHECK-1SCOPE:       %[[RESULT:.+]] = pcf.generic
// CHECK-1SCOPE-SAME:    scope(#pcf.sequential)
// CHECK-1SCOPE:         execute(%[[REF:.+]] = %[[INIT]])[%[[ID:.+]]: index, %[[COUNT:.+]]: index]
// Tile size computed from total / count.
// CHECK-1SCOPE:         %[[TILE_SIZE:.+]] = arith.ceildivui %{{.+}}, %[[COUNT]]
// Bounds: start = id * tile_size.
// CHECK-1SCOPE:         %[[START:.+]] = arith.muli %[[ID]], %[[TILE_SIZE]]
// CHECK-1SCOPE:         %[[END_UNCLAMPED:.+]] = arith.addi %[[START]], %[[TILE_SIZE]]
// CHECK-1SCOPE:         %[[END:.+]] = arith.minui %[[END_UNCLAMPED]]
// Forall from start to end.
// CHECK-1SCOPE:         scf.forall (%[[IV:.+]]) = (%[[START]]) to (%[[END]])
// CHECK-1SCOPE:           %[[SLICE:.+]] = tensor.extract_slice %[[INIT]][%[[IV]]]
// CHECK-1SCOPE:           pcf.write_slice %[[SLICE]] into %[[REF]][%[[IV]]]
// CHECK-1SCOPE:         pcf.return
// CHECK-1SCOPE:       return %[[RESULT]]

// CHECK-2SCOPE-LABEL: func.func @test_1d_single_scope
// CHECK-2SCOPE-SAME:    %[[INIT:.+]]: tensor<64xf32>
// CHECK-2SCOPE:       %[[RESULT:.+]] = pcf.generic
// CHECK-2SCOPE-SAME:    scope(#pcf.sequential)
// CHECK-2SCOPE:         execute(%[[REF:.+]] = %[[INIT]])[%[[ID0:.+]]: index, %[[COUNT0:.+]]: index]
// CHECK-2SCOPE:         pcf.generic
// CHECK-2SCOPE-SAME:      scope(#pcf.sequential)
// CHECK-2SCOPE:           execute[%[[ID1:.+]]: index, %[[COUNT1:.+]]: index]
// Linearize the two scope IDs.
// CHECK-2SCOPE:           %[[LIN_ID:.+]] = affine.linearize_index [%[[ID0]], %[[ID1]]] by (%[[COUNT0]], %[[COUNT1]])
// CHECK-2SCOPE:           %[[TOTAL_COUNT:.+]] = arith.muli %[[COUNT0]], %[[COUNT1]]
// Tile size computed from total / linearized_count.
// CHECK-2SCOPE:           %[[TILE_SIZE:.+]] = arith.ceildivui %{{.+}}, %[[TOTAL_COUNT]]
// Bounds using linearized ID.
// CHECK-2SCOPE:           %[[START:.+]] = arith.muli %[[LIN_ID]], %[[TILE_SIZE]]
// CHECK-2SCOPE:           %[[END_UNCLAMPED:.+]] = arith.addi %[[START]], %[[TILE_SIZE]]
// CHECK-2SCOPE:           %[[END:.+]] = arith.minui %[[END_UNCLAMPED]]
// Forall from start to end.
// CHECK-2SCOPE:           scf.forall (%[[IV:.+]]) = (%[[START]]) to (%[[END]])
// CHECK-2SCOPE:             pcf.write_slice %{{.+}} into %[[REF]][%[[IV]]]
// CHECK-2SCOPE:           pcf.return
// CHECK-2SCOPE:         pcf.return
// CHECK-2SCOPE:       return %[[RESULT]]
func.func @test_1d_single_scope(%init: tensor<64xf32>) -> tensor<64xf32> {
  %result = scf.forall (%i) in (64) shared_outs(%out = %init) -> tensor<64xf32> {
    %slice = tensor.extract_slice %out[%i] [1] [1] : tensor<64xf32> to tensor<1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %out[%i] [1] [1] : tensor<1xf32> into tensor<64xf32>
    }
  } {mapping = [#iree_codegen.local_mapping<0>]}
  return %result : tensor<64xf32>
}

// -----

// CHECK-1SCOPE-LABEL: func.func @test_2d_multi_dim
// CHECK-1SCOPE-SAME:    %[[INIT:.+]]: tensor<64x128xf32>
// CHECK-1SCOPE:       pcf.generic
// CHECK-1SCOPE:         execute(%[[REF:.+]] = %[[INIT]])[%[[ID:.+]]: index, %[[COUNT:.+]]: index]
// Tile size from total elements / count.
// CHECK-1SCOPE:         %[[TILE_SIZE:.+]] = arith.ceildivui %{{.+}}, %[[COUNT]]
// CHECK-1SCOPE:         %[[START:.+]] = arith.muli %[[ID]], %[[TILE_SIZE]]
// CHECK-1SCOPE:         %[[END_UNCLAMPED:.+]] = arith.addi %[[START]], %[[TILE_SIZE]]
// CHECK-1SCOPE:         %[[END:.+]] = arith.minui %[[END_UNCLAMPED]]
// CHECK-1SCOPE:         scf.forall (%[[IV:.+]]) = (%[[START]]) to (%[[END]])
// Delinearize into (64, 128) to recover 2D indices.
// CHECK-1SCOPE:           %[[INDICES:.+]]:2 = affine.delinearize_index %[[IV]] into (64, 128)
// CHECK-1SCOPE:           %[[SLICE:.+]] = tensor.extract_slice %[[INIT]][%[[INDICES]]#0, %[[INDICES]]#1]
// CHECK-1SCOPE:           pcf.write_slice %[[SLICE]] into %[[REF]][%[[INDICES]]#0, %[[INDICES]]#1]
// CHECK-1SCOPE:         pcf.return

// CHECK-2SCOPE-LABEL: func.func @test_2d_multi_dim
// CHECK-2SCOPE:       pcf.generic
// CHECK-2SCOPE-SAME:    scope(#pcf.sequential)
// CHECK-2SCOPE:         execute(%[[REF:.+]] = %{{.+}})[%[[ID0:.+]]: index, %[[COUNT0:.+]]: index]
// CHECK-2SCOPE:         pcf.generic
// CHECK-2SCOPE-SAME:      scope(#pcf.sequential)
// CHECK-2SCOPE:           execute[%[[ID1:.+]]: index, %[[COUNT1:.+]]: index]
// CHECK-2SCOPE:           %[[LIN_ID:.+]] = affine.linearize_index [%[[ID0]], %[[ID1]]] by (%[[COUNT0]], %[[COUNT1]])
// CHECK-2SCOPE:           %[[TOTAL_COUNT:.+]] = arith.muli %[[COUNT0]], %[[COUNT1]]
// CHECK-2SCOPE:           scf.forall (%[[IV:.+]]) =
// Delinearize the forall IV to recover 2D indices.
// CHECK-2SCOPE:             %[[INDICES:.+]]:2 = affine.delinearize_index %[[IV]] into (64, 128)
// CHECK-2SCOPE:             pcf.write_slice %{{.+}} into %[[REF]][%[[INDICES]]#0, %[[INDICES]]#1]
// CHECK-2SCOPE:           pcf.return
// CHECK-2SCOPE:         pcf.return
func.func @test_2d_multi_dim(%init: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %result = scf.forall (%i, %j) in (64, 128) shared_outs(%out = %init) -> tensor<64x128xf32> {
    %slice = tensor.extract_slice %out[%i, %j] [1, 1] [1, 1] : tensor<64x128xf32> to tensor<1x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %out[%i, %j] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<64x128xf32>
    }
  } {mapping = [#iree_codegen.local_mapping<1>, #iree_codegen.local_mapping<0>]}
  return %result : tensor<64x128xf32>
}

// -----

// Test mapping permutation where dimensions are NOT in natural order.
// Forall dimensions: (%i:4, %j:8) with mapping [local_mapping<0>, local_mapping<1>].
// This means: dim 0 (i) corresponds to id 0 (fast), dim 1 (j) to id 1 (slow).
// Linearization should be: j * 4 + i (not i * 8 + j).
// Delinearization with basis (8, 4) gives (j, i) which must be permuted.

// CHECK-1SCOPE-LABEL: func.func @test_permutation
// CHECK-1SCOPE-SAME:    %[[INIT:.+]]: tensor<4x8xf32>
// CHECK-1SCOPE:       pcf.generic
// CHECK-1SCOPE:         execute(%[[REF:.+]] = %[[INIT]])[%[[ID:.+]]: index, %[[COUNT:.+]]: index]
// CHECK-1SCOPE:         scf.forall (%[[IV:.+]]) =
// Delinearization basis should be (8, 4) - slow dimension first.
// CHECK-1SCOPE:           %[[INDICES:.+]]:2 = affine.delinearize_index %[[IV]] into (8, 4)
// Permuted indices: [#1, #0] maps (j, i) back to (i, j) for tensor access.
// CHECK-1SCOPE:           %[[SLICE:.+]] = tensor.extract_slice %[[INIT]][%[[INDICES]]#1, %[[INDICES]]#0]
// CHECK-1SCOPE:           pcf.write_slice %[[SLICE]] into %[[REF]][%[[INDICES]]#1, %[[INDICES]]#0]
// CHECK-1SCOPE:         pcf.return
func.func @test_permutation(%init: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %result = scf.forall (%i, %j) in (4, 8) shared_outs(%out = %init) -> tensor<4x8xf32> {
    %slice = tensor.extract_slice %out[%i, %j] [1, 1] [1, 1] : tensor<4x8xf32> to tensor<1x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %out[%i, %j] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<4x8xf32>
    }
  } {mapping = [#iree_codegen.local_mapping<0>, #iree_codegen.local_mapping<1>]}
  return %result : tensor<4x8xf32>
}

// -----

// Test 3D local_mapping with permutation [1, 2, 0].
// This means: dim0 -> position 1, dim1 -> position 2, dim2 -> position 0.
// So iteration order from fastest to slowest is: dim2, dim0, dim1.
// Bounds are (4, 8, 16), so delinearization basis should be (8, 4, 16)
// ordered from slowest to fastest: dim1 (8), dim0 (4), dim2 (16).

// CHECK-1SCOPE-LABEL: func.func @test_3d_permutation
// CHECK-1SCOPE-SAME:    %[[INIT:.+]]: tensor<4x8x16xf32>
// CHECK-1SCOPE:       pcf.generic
// CHECK-1SCOPE:         execute(%[[REF:.+]] = %[[INIT]])[%[[ID:.+]]: index, %[[COUNT:.+]]: index]
// CHECK-1SCOPE:         scf.forall (%[[IV:.+]]) =
// Delinearization basis should be (8, 4, 16) - slowest to fastest.
// CHECK-1SCOPE:           %[[INDICES:.+]]:3 = affine.delinearize_index %[[IV]] into (8, 4, 16)
// Permuted indices: [#1, #0, #2] maps back to original dim order.
// CHECK-1SCOPE:           %[[SLICE:.+]] = tensor.extract_slice %[[INIT]][%[[INDICES]]#1, %[[INDICES]]#0, %[[INDICES]]#2]
// CHECK-1SCOPE:           pcf.write_slice %[[SLICE]] into %[[REF]][%[[INDICES]]#1, %[[INDICES]]#0, %[[INDICES]]#2]
// CHECK-1SCOPE:         pcf.return
func.func @test_3d_permutation(%init: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %result = scf.forall (%i, %j, %k) in (4, 8, 16) shared_outs(%out = %init) -> tensor<4x8x16xf32> {
    %slice = tensor.extract_slice %out[%i, %j, %k] [1, 1, 1] [1, 1, 1] : tensor<4x8x16xf32> to tensor<1x1x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %out[%i, %j, %k] [1, 1, 1] [1, 1, 1] : tensor<1x1x1xf32> into tensor<4x8x16xf32>
    }
  } {mapping = [#iree_codegen.local_mapping<1>, #iree_codegen.local_mapping<2>, #iree_codegen.local_mapping<0>]}
  return %result : tensor<4x8x16xf32>
}

// -----

// CHECK-1SCOPE-LABEL: func.func @test_empty_mapping
// CHECK-1SCOPE-SAME:    %[[INIT:.+]]: tensor<64xf32>
// CHECK-1SCOPE:       pcf.generic
// CHECK-1SCOPE:         execute(%[[REF:.+]] = %[[INIT]])[%[[ID:.+]]: index, %[[COUNT:.+]]: index]
// CHECK-1SCOPE:         scf.forall (%[[IV:.+]]) =
// CHECK-1SCOPE:           %[[SLICE:.+]] = tensor.extract_slice %[[INIT]][%[[IV]]]
// CHECK-1SCOPE:           pcf.write_slice %[[SLICE]] into %[[REF]][%[[IV]]]
// CHECK-1SCOPE:         pcf.return
func.func @test_empty_mapping(%init: tensor<64xf32>) -> tensor<64xf32> {
  %result = scf.forall (%i) in (64) shared_outs(%out = %init) -> tensor<64xf32> {
    %slice = tensor.extract_slice %out[%i] [1] [1] : tensor<64xf32> to tensor<1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %out[%i] [1] [1] : tensor<1xf32> into tensor<64xf32>
    }
  }
  return %result : tensor<64xf32>
}
