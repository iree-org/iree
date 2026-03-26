// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-im2col{unroll=false}, canonicalize, cse))" --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-im2col{unroll=true}))" --split-input-file %s | FileCheck %s --check-prefix=CHECK-UNROLL

// Test 1: Dynamic M offset and K offset.
// The decomposition produces two nested scf.for loops (batch, M), computes
// spatial coords via affine.apply, then extract_slice + linalg.copy.
#map = affine_map<(d0) -> (d0 * 4)>
module {
  func.func @im2col_untile_k(%arg0: tensor<2x34x34x640xf32>, %m_size: index, %m_off: index, %k: index) -> tensor<2x?x4xf32> {
    %0 = tensor.empty(%m_size) : tensor<2x?x4xf32>
    %k_off = affine.apply #map(%k)
    %7 = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
            offsets = [0, %m_off, %k_off] output_sizes = [[2], [32, 32], [3, 3, 640]]
            batch_pos = [0] m_pos = [1, 2] k_pos = [3]
            input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
            ins(%arg0 : tensor<2x34x34x640xf32>)
            outs(%0 : tensor<2x?x4xf32>) -> tensor<2x?x4xf32>
    return %7 : tensor<2x?x4xf32>
  }
}
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 * 4)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-LABEL: func.func @im2col_untile_k
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[mSIZE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[mOFF:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[K:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[OUT_TILE:.+]] = tensor.empty(%[[mSIZE]]) : tensor<2x?x4xf32>
//       CHECK:   %[[bLOOP:.+]] = scf.for %[[b:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUT0:.+]] = %[[OUT_TILE]]) -> (tensor<2x?x4xf32>)
//       CHECK:     %[[mLOOP:.+]] = scf.for %[[m:.+]] = %[[C0]] to %[[mSIZE]] step %[[C1]] iter_args(%[[OUT1:.+]] = %[[OUT0]]) -> (tensor<2x?x4xf32>)
//   CHECK-DAG:       %[[kScaled:.+]] = affine.apply #[[$MAP]]()[%[[K]]]
//   CHECK-DAG:       %[[kParts:.+]]:3 = affine.delinearize_index %[[kScaled]] into (3, 3, 640)
//   CHECK-DAG:       %[[mIDX:.+]] = affine.apply #[[$MAP1]](%[[m]])[%[[mOFF]]]
//   CHECK-DAG:       %[[mParts:.+]]:2 = affine.delinearize_index %[[mIDX]] into (32, 32)
//   CHECK-DAG:       %[[h:.+]] = affine.apply #[[$MAP1]](%[[mParts]]#0)[%[[kParts]]#0]
//   CHECK-DAG:       %[[w:.+]] = affine.apply #[[$MAP1]](%[[mParts]]#1)[%[[kParts]]#1]
//       CHECK:       %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[b]], %[[h]], %[[w]], %[[kParts]]#2] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<2x34x34x640xf32> to tensor<1x1x4xf32>
//       CHECK:       %[[OUT_SLICE:.+]] = tensor.extract_slice %[[OUT1]][%[[b]], %[[m]], 0] [1, 1, 4] [1, 1, 1] : tensor<2x?x4xf32> to tensor<1x1x4xf32>
//       CHECK:       %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<1x1x4xf32>) outs(%[[OUT_SLICE]] : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
//       CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[OUT1]][%[[b]], %[[m]], 0] [1, 1, 4] [1, 1, 1] : tensor<1x1x4xf32> into tensor<2x?x4xf32>
//       CHECK:       scf.yield %[[INSERT]] : tensor<2x?x4xf32>
//       CHECK:     scf.yield %[[mLOOP]] : tensor<2x?x4xf32>
//       CHECK:   return %[[bLOOP]] : tensor<2x?x4xf32>
// CHECK-UNROLL-LABEL: func.func @im2col_untile_k
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
// Verify two unrolled batch loops (b=0 and b=1), each iterating over M.
//       CHECK-UNROLL:   scf.for %{{.+}} = %{{.+}} to %[[mS:.+]] step
//       CHECK-UNROLL:   linalg.copy
//       CHECK-UNROLL:   scf.for %{{.+}} = %{{.+}} to %[[mS]] step
//       CHECK-UNROLL:   linalg.copy

// -----

// Test 2: Dynamic M and K offsets with transposed m_pos.
// Three nested loops (batch, M, K). Spatial coords are computed via affine.apply
// using strides and dilations.
module {
  func.func @im2col_transposed_m_pos(%arg0: tensor<640x2x101x172xf32>, %m_size: index, %k_size: index, %m_off: index, %k_off: index) -> tensor<2x?x?xf32> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty(%m_size, %k_size) : tensor<2x?x?xf32>
    %8 = iree_linalg_ext.im2col
            strides = [5, 3] dilations = [4, 7] kernel_size = [5, 2]
            offsets = [0, %m_off, %k_off] output_sizes = [[2], [32, 32], [640, 5, 2]]
            batch_pos = [1] m_pos = [3, 2] k_pos = [0]
            input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
            ins(%arg0 : tensor<640x2x101x172xf32>)
            outs(%0 : tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
    return %8 : tensor<2x?x?xf32>
  }
}
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0 * 5 + d1 * 4)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1) -> (d0 * 3 + d1 * 7)>
// CHECK-LABEL: func.func @im2col_transposed_m_pos
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[mSIZE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[kSIZE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[mOFF:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[kOFF:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[OUT_TILE:.+]] = tensor.empty(%[[mSIZE]], %[[kSIZE]]) : tensor<2x?x?xf32>
//       CHECK:   %[[bLOOP:.+]] = scf.for %[[b:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUT0:.+]] = %[[OUT_TILE]]) -> (tensor<2x?x?xf32>)
//       CHECK:     %[[mLOOP:.+]] = scf.for %[[m:.+]] = %[[C0]] to %[[mSIZE]] step %[[C1]] iter_args(%[[OUT1:.+]] = %[[OUT0]]) -> (tensor<2x?x?xf32>)
//       CHECK:       %[[kLOOP:.+]] = scf.for %[[k:.+]] = %[[C0]] to %[[kSIZE]] step %[[C1]] iter_args(%[[OUT2:.+]] = %[[OUT1]]) -> (tensor<2x?x?xf32>)
//   CHECK-DAG:         %[[kIDX:.+]] = affine.apply #[[$MAP]](%[[k]])[%[[kOFF]]]
//   CHECK-DAG:         %[[kParts:.+]]:3 = affine.delinearize_index %[[kIDX]] into (640, 5, 2)
//   CHECK-DAG:         %[[mIDX:.+]] = affine.apply #[[$MAP]](%[[m]])[%[[mOFF]]]
//   CHECK-DAG:         %[[mParts:.+]]:2 = affine.delinearize_index %[[mIDX]] into (32, 32)
//   CHECK-DAG:         affine.apply #[[$MAP1]](%[[mParts]]#0, %[[kParts]]#1)
//   CHECK-DAG:         affine.apply #[[$MAP2]](%[[mParts]]#1, %[[kParts]]#2)
//       CHECK:         %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[kParts]]#0, %[[b]], {{.*}}, {{.*}}] [1, 1, 1, 1]
//       CHECK:         %[[OUT_SLICE:.+]] = tensor.extract_slice %[[OUT2]][%[[b]], %[[m]], %[[k]]] [1, 1, 1] [1, 1, 1] : tensor<2x?x?xf32> to tensor<1x1x1xf32>
//       CHECK:         linalg.copy ins(%[[IN_SLICE]] : tensor<1x1x1xf32>) outs(%[[OUT_SLICE]] : tensor<1x1x1xf32>)
//       CHECK:         tensor.insert_slice {{.*}} into %[[OUT2]][%[[b]], %[[m]], %[[k]]] [1, 1, 1] [1, 1, 1] : tensor<1x1x1xf32> into tensor<2x?x?xf32>
//       CHECK:         scf.yield {{.*}} : tensor<2x?x?xf32>
//       CHECK:       scf.yield %[[kLOOP]] : tensor<2x?x?xf32>
//       CHECK:     scf.yield %[[mLOOP]] : tensor<2x?x?xf32>
//       CHECK:   return %[[bLOOP]] : tensor<2x?x?xf32>

// CHECK-UNROLL-LABEL: func.func @im2col_transposed_m_pos
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
// Two unrolled copies of the m/k loops for batch=0 and batch=1.
//       CHECK-UNROLL:   scf.for
//       CHECK-UNROLL:   linalg.copy
//       CHECK-UNROLL:   scf.for
//       CHECK-UNROLL:   linalg.copy

// -----

// Test 3: Static sizes with expanded M and K output dims.
// Four nested loops (batch, M0, M1, K). Spatial coords via affine.apply.
module {
  func.func @im2col_expanded(%arg0: tensor<2x34x34x640xf32>, %m_size0: index, %m_size1: index, %m0: index, %m1: index, %k: index, %m_stride: index) -> tensor<2x?x?x2x4xf32> {
    %0 = tensor.empty(%m_size0, %m_size1) : tensor<2x?x?x2x4xf32>
    %7 = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
            offsets = [0, %m0, %m1, %k, 0] output_sizes = [[2], [32], [32], [3, 3], [640]]
            batch_pos = [0] m_pos = [1, 2] k_pos = [3]
            input_k_perm = [0, 1, 2] output_perm = [0, 1, 2, 3, 4]
            ins(%arg0 : tensor<2x34x34x640xf32>)
            outs(%0 : tensor<2x?x?x2x4xf32>) -> tensor<2x?x?x2x4xf32>
    return %7 : tensor<2x?x?x2x4xf32>
  }
}
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1)[s0] -> (d0 + d1 + s0)>
// CHECK-LABEL: func.func @im2col_expanded
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[mSIZE0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[mSIZE1:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[mOFF0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[mOFF1:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[kOFF:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[OUT_TILE:.+]] = tensor.empty(%[[mSIZE0]], %[[mSIZE1]]) : tensor<2x?x?x2x4xf32>
//       CHECK:   %[[bLOOP:.+]] = scf.for %[[b:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUT0:.+]] = %[[OUT_TILE]]) -> (tensor<2x?x?x2x4xf32>)
//       CHECK:     %[[mLOOP0:.+]] = scf.for %[[m0:.+]] = %[[C0]] to %[[mSIZE0]] step %[[C1]] iter_args(%[[OUT1:.+]] = %[[OUT0]]) -> (tensor<2x?x?x2x4xf32>)
//       CHECK:       %[[mLOOP1:.+]] = scf.for %[[m1:.+]] = %[[C0]] to %[[mSIZE1]] step %[[C1]] iter_args(%[[OUT2:.+]] = %[[OUT1]]) -> (tensor<2x?x?x2x4xf32>)
//       CHECK:         %[[kLOOP:.+]] = scf.for %[[k:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUT3:.+]] = %[[OUT2]]) -> (tensor<2x?x?x2x4xf32>)
//   CHECK-DAG:           %[[kPOS:.+]] = affine.apply #[[$MAP]](%[[k]])[%[[kOFF]]]
//   CHECK-DAG:           %[[kParts:.+]]:2 = affine.delinearize_index %[[kPOS]] into (3, 3)
//   CHECK-DAG:           affine.apply #[[$MAP1]](%[[kParts]]#0, %[[m0]])[%[[mOFF0]]]
//   CHECK-DAG:           affine.apply #[[$MAP1]](%[[kParts]]#1, %[[m1]])[%[[mOFF1]]]
//       CHECK:           %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[b]], {{.*}}, {{.*}}, 0] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<2x34x34x640xf32> to tensor<1x1x1x4xf32>
//       CHECK:           %[[OUT_SLICE:.+]] = tensor.extract_slice %[[OUT3]][%[[b]], %[[m0]], %[[m1]], %[[k]], 0] [1, 1, 1, 1, 4] [1, 1, 1, 1, 1] : tensor<2x?x?x2x4xf32> to tensor<1x1x1x4xf32>
//       CHECK:           %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<1x1x1x4xf32>) outs(%[[OUT_SLICE]] : tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
//       CHECK:           %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[OUT3]][%[[b]], %[[m0]], %[[m1]], %[[k]], 0] [1, 1, 1, 1, 4] [1, 1, 1, 1, 1] : tensor<1x1x1x4xf32> into tensor<2x?x?x2x4xf32>
//       CHECK:           scf.yield %[[INSERT]] : tensor<2x?x?x2x4xf32>
//       CHECK:         scf.yield %[[kLOOP]] : tensor<2x?x?x2x4xf32>
//       CHECK:       scf.yield %[[mLOOP1]] : tensor<2x?x?x2x4xf32>
//       CHECK:     scf.yield %[[mLOOP0]] : tensor<2x?x?x2x4xf32>
//       CHECK:   return %[[bLOOP]] : tensor<2x?x?x2x4xf32>

// CHECK-UNROLL-LABEL: func.func @im2col_expanded
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
// Unrolled: batch loop is removed, m0/m1 loops remain; K=2 is unrolled in each body.
//       CHECK-UNROLL:   scf.for
//       CHECK-UNROLL:     scf.for
//       CHECK-UNROLL:       linalg.copy
//       CHECK-UNROLL:       linalg.copy

// -----

// Test 4: NCHW layout with static sizes -- scalar fallback (34 % 4 != 0).
// Three nested loops (batch, K-row, K-col-channel) with scalar linalg.copy.
module {
  func.func @im2col_expanded_nchw(%arg0: tensor<2x640x34x34xf32>, %m0: index, %m1: index, %k: index) -> tensor<2x1x1x2x4xf32> {
    %0 = tensor.empty() : tensor<2x1x1x2x4xf32>
    %7 = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
            offsets = [0, %m0, %m1, %k, 0] output_sizes = [[2], [638], [32], [3, 3], [34]]
            batch_pos = [0] m_pos = [1, 2] k_pos = [3]
            input_k_perm = [0, 1, 2] output_perm = [0, 1, 2, 3, 4]
            ins(%arg0 : tensor<2x640x34x34xf32>)
            outs(%0 : tensor<2x1x1x2x4xf32>) -> tensor<2x1x1x2x4xf32>
    return %7 : tensor<2x1x1x2x4xf32>
  }
}
// Scalar fallback (34 % 4 != 0): three nested loops with scalar linalg.copy.
// CHECK-LABEL: func.func @im2col_expanded_nchw
//       CHECK:   tensor.empty() : tensor<2x1x1x2x4xf32>
//       CHECK:   scf.for
//       CHECK:     scf.for
//       CHECK:       scf.for
//       CHECK:         affine.delinearize_index {{.*}} into (3, 3)
//       CHECK:         tensor.extract_slice {{.*}} : tensor<2x640x34x34xf32> to tensor<1x1x1x1xf32>
//       CHECK:         linalg.copy ins({{.*}} : tensor<1x1x1x1xf32>) outs({{.*}} : tensor<1x1x1x1xf32>)
//       CHECK:         tensor.insert_slice {{.*}} : tensor<1x1x1x1xf32> into tensor<2x1x1x2x4xf32>

// CHECK-UNROLL-LABEL: func.func @im2col_expanded_nchw
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col

// -----

// Test 5: Backward-weight-style im2col with dilation=2, single-element M and K dims.
// Directly extract_slices at (m0, m1) without any loops (output is 1x1x1x1).
module {
  func.func @im2col_bwd_weight_dilation2(%arg0: tensor<1x1x8x8xf32>, %m0: index, %m1: index) -> tensor<1x1x1x1xf32> {
    %0 = tensor.empty() : tensor<1x1x1x1xf32>
    %result = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [2, 2] kernel_size = [3, 3]
            offsets = [0, %m0, %m1, 0] output_sizes = [[1], [3], [3], [1, 3, 3]]
            batch_pos = [0] m_pos = [2, 3] k_pos = [1]
            input_k_perm = [0, 1, 2] output_perm = [0, 1, 2, 3]
            ins(%arg0 : tensor<1x1x8x8xf32>)
            outs(%0 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    return %result : tensor<1x1x1x1xf32>
  }
}
// With single-element output (1x1x1x1), emits a bare
// extract_slice + linalg.copy without any loops.
// CHECK-LABEL: func.func @im2col_bwd_weight_dilation2
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1x8x8xf32>
//  CHECK-SAME:     %[[M0:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[M1:[a-zA-Z0-9_]+]]: index
//       CHECK:   tensor.extract_slice %[[ARG0]][0, 0, %[[M0]], %[[M1]]]
//       CHECK:   linalg.copy
//       CHECK:   return
//   CHECK-NOT:   iree_linalg_ext.im2col

// CHECK-UNROLL-LABEL: func.func @im2col_bwd_weight_dilation2
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
// Fully unrolled 1x1x1x1 output: bare extract_slice + copy, no loops.
//       CHECK-UNROLL:   tensor.extract_slice
//       CHECK-UNROLL:   linalg.copy
//       CHECK-UNROLL:   return

// -----

// Test 6: Static sizes with dynamic M offset, with unrolling.
// The unrolled pass unrolls the static batch (size 2) and M (size 2) dims into
// separate extract_slice + linalg.copy + insert_slice blocks.
#map6 = affine_map<(d0) -> (d0 * 4)>
module {
  func.func @im2col_unrolled(%arg0: tensor<2x34x34x640xf32>, %m_off: index, %k: index) -> tensor<2x2x4xf32> {
    %0 = tensor.empty() : tensor<2x2x4xf32>
    %k_off = affine.apply #map6(%k)
    %7 = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
            offsets = [0, %m_off, %k_off] output_sizes = [[2], [32, 32], [3, 3, 640]]
            batch_pos = [0] m_pos = [1, 2] k_pos = [3]
            input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
            ins(%arg0 : tensor<2x34x34x640xf32>)
            outs(%0 : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
    return %7 : tensor<2x2x4xf32>
  }
}
// CHECK-LABEL: func.func @im2col_unrolled
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[mOFF:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[K:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[OUT_TILE:.+]] = tensor.empty() : tensor<2x2x4xf32>
// Non-unrolled output uses two nested loops (batch=2, M=2).
//       CHECK:   %[[bLOOP:.+]] = scf.for %[[b:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUT0:.+]] = %[[OUT_TILE]]) -> (tensor<2x2x4xf32>)
//       CHECK:     %[[mLOOP:.+]] = scf.for %[[m:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUT1:.+]] = %[[OUT0]]) -> (tensor<2x2x4xf32>)
//       CHECK:       tensor.extract_slice %[[ARG0]]
//       CHECK:       linalg.copy ins({{.*}} : tensor<1x1x4xf32>) outs({{.*}} : tensor<1x1x4xf32>)
//       CHECK:       tensor.insert_slice {{.*}} into %[[OUT1]][%[[b]], %[[m]], 0] [1, 1, 4] [1, 1, 1] : tensor<1x1x4xf32> into tensor<2x2x4xf32>
//       CHECK:       scf.yield {{.*}} : tensor<2x2x4xf32>
//       CHECK:     scf.yield %[[mLOOP]] : tensor<2x2x4xf32>
//       CHECK:   return %[[bLOOP]] : tensor<2x2x4xf32>
//   CHECK-NOT:   iree_linalg_ext.im2col

// CHECK-UNROLL-LABEL: func.func @im2col_unrolled
//  CHECK-UNROLL-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]
//  CHECK-UNROLL-SAME:     %[[mOFF:[a-zA-Z0-9_]+]]
//  CHECK-UNROLL-SAME:     %[[K:[a-zA-Z0-9_]+]]
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
//       CHECK-UNROLL:   %[[EMPTY:.*]] = tensor.empty() : tensor<2x2x4xf32>
// Unrolled: 4 copies (b=0,m=0), (b=0,m=1), (b=1,m=0), (b=1,m=1).
//       CHECK-UNROLL:   linalg.copy ins({{.*}} : tensor<1x1x4xf32>) outs({{.*}} : tensor<1x1x4xf32>)
//       CHECK-UNROLL:   %[[INS0:.*]] = tensor.insert_slice %{{.*}} into %[[EMPTY]]
//       CHECK-UNROLL:   linalg.copy ins({{.*}} : tensor<1x1x4xf32>) outs({{.*}} : tensor<1x1x4xf32>)
//       CHECK-UNROLL:   %[[INS1:.*]] = tensor.insert_slice %{{.*}} into %[[INS0]]
//       CHECK-UNROLL:   linalg.copy ins({{.*}} : tensor<1x1x4xf32>) outs({{.*}} : tensor<1x1x4xf32>)
//       CHECK-UNROLL:   %[[INS2:.*]] = tensor.insert_slice %{{.*}} into %[[INS1]]
//       CHECK-UNROLL:   linalg.copy ins({{.*}} : tensor<1x1x4xf32>) outs({{.*}} : tensor<1x1x4xf32>)
//       CHECK-UNROLL:   %[[INS3:.*]] = tensor.insert_slice %{{.*}} into %[[INS2]]
//       CHECK-UNROLL:   return %[[INS3]] : tensor<2x2x4xf32>

// -----

// Test 7: im2col with pre-padded input (tensor.pad before im2col).
// This uses the padding-aware code path which does have affine.max/affine.min ops.
module {
  func.func @im2col_padding(%input: tensor<1x8x3x3xf32>) -> tensor<1x2x2x12xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<1x2x2x12xf32>
    %padded = tensor.pad %input low[0, 0, 3, 3] high[0, 0, 3, 3] {
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
    tensor.yield %cst : f32
  } : tensor<1x8x3x3xf32> to tensor<1x8x9x9xf32>
  %im2col = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
                              offsets = [0, 0, 0, 0] output_sizes = [[1], [7], [7], [8, 3, 3]]
                              batch_pos = [0] m_pos = [2, 3] k_pos = [1]
                              input_k_perm = [0, 1, 2] output_perm = [0, 1, 2, 3]
                              ins(%padded : tensor<1x8x9x9xf32>)
                              outs(%empty : tensor<1x2x2x12xf32>) -> tensor<1x2x2x12xf32>
  return %im2col : tensor<1x2x2x12xf32>
  }
}

// The im2col_padding test has a complex output with affine.max/affine.min ops
// due to the pre-padded input. Use simpler CHECK patterns.
// CHECK-LABEL: func.func @im2col_padding
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]
//       CHECK: tensor.extract_slice %[[ARG0]]
//       CHECK: tensor.pad
//  CHECK-NEXT: ^bb0
//  CHECK-NEXT:   tensor.yield
//       CHECK: linalg.copy
//       CHECK: tensor.insert_slice
//   CHECK-NOT: iree_linalg_ext.im2col

// CHECK-UNROLL-LABEL: func.func @im2col_padding
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
//       CHECK-UNROLL:   tensor.pad
//       CHECK-UNROLL:   linalg.copy

// -----

// Test 8: Static sizes, NHWC layout with non-identity input_k_perm -- scalar fallback.
// The non-identity k_perm prevents vectorization; two nested loops (M, K) with scalar copy.
module {
  func.func @im2col_nhc_with_perm(%arg0: tensor<1x3x2xf32>) -> tensor<1x2x4xf32> {
    %0 = tensor.empty() : tensor<1x2x4xf32>
    %1 = iree_linalg_ext.im2col strides = [1] dilations = [1] kernel_size = [2]
                            offsets = [0, 0, 0] output_sizes = [[1], [2], [2, 2]]
                            batch_pos = [0] m_pos = [1] k_pos = [2]
                            input_k_perm = [1, 0] output_perm = [0, 1, 2]
                            ins(%arg0 : tensor<1x3x2xf32>)
                            outs(%0 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
    return %1 : tensor<1x2x4xf32>
  }
}
// Scalar fallback with k_perm: 2 loops (M + K) with linalg.copy.
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: func.func @im2col_nhc_with_perm
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x3x2xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//       CHECK:   %[[OUT_TILE:.+]] = tensor.empty() : tensor<1x2x4xf32>
//       CHECK:   %[[MLOOP:.+]] = scf.for %[[M:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUT1:.+]] = %[[OUT_TILE]]) -> (tensor<1x2x4xf32>)
//       CHECK:     %[[KLOOP:.+]] = scf.for %[[K:.+]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[OUT2:.+]] = %[[OUT1]]) -> (tensor<1x2x4xf32>)
//       CHECK:       %[[kParts:.+]]:2 = affine.delinearize_index %[[K]] into (2, 2) : index, index
//       CHECK:       %[[hIDX:.+]] = affine.apply #[[$MAP]](%[[M]], %[[kParts]]#1)
//       CHECK:       %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[hIDX]], %[[kParts]]#0] [1, 1, 1] [1, 1, 1] : tensor<1x3x2xf32> to tensor<1x1x1xf32>
//       CHECK:       %[[OUT_SLICE:.+]] = tensor.extract_slice %[[OUT2]][0, %[[M]], %[[K]]] [1, 1, 1] [1, 1, 1] : tensor<1x2x4xf32> to tensor<1x1x1xf32>
//       CHECK:       %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<1x1x1xf32>) outs(%[[OUT_SLICE]] : tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
//       CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[OUT2]][0, %[[M]], %[[K]]] [1, 1, 1] [1, 1, 1] : tensor<1x1x1xf32> into tensor<1x2x4xf32>
//       CHECK:       scf.yield %[[INSERT]] : tensor<1x2x4xf32>
//       CHECK:     scf.yield %[[KLOOP]] : tensor<1x2x4xf32>
//       CHECK:   return %[[MLOOP]] : tensor<1x2x4xf32>

// CHECK-UNROLL-LABEL: func.func @im2col_nhc_with_perm
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
//       CHECK-UNROLL:   linalg.copy

// -----

// Test 9: Non-self-inverse input_k_perm = [2, 0, 1].
// The inverse permutation [1, 2, 0] maps output K coords (ch, kH, kW) -> input
// order (kH, kW, ch). This is now decomposed as a scalar fallback.
module {
  func.func @im2col_nhwc_with_perm(%arg0: tensor<1x16x16x4xf32>) -> tensor<1x14x14x36xf32> {
    %0 = tensor.empty() : tensor<1x14x14x36xf32>
    %1 = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
                            offsets = [0, 0, 0, 0] output_sizes = [[1], [14], [14], [4, 3, 3]]
                            batch_pos = [0] m_pos = [1, 2] k_pos = [3]
                            input_k_perm = [2, 0, 1] output_perm = [0, 1, 2, 3]
                            ins(%arg0 : tensor<1x16x16x4xf32>)
                            outs(%0 : tensor<1x14x14x36xf32>) -> tensor<1x14x14x36xf32>
    return %1 : tensor<1x14x14x36xf32>
  }
}
// Scalar fallback with non-self-inverse kperm: loops over M0, M1, K.
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: func.func @im2col_nhwc_with_perm
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x16x16x4xf32>
//   CHECK-DAG:   %[[C36:.+]] = arith.constant 36 : index
//   CHECK-DAG:   %[[C14:.+]] = arith.constant 14 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[OUT_TILE:.+]] = tensor.empty() : tensor<1x14x14x36xf32>
//       CHECK:   %[[MLOOP0:.+]] = scf.for %[[M0:.+]] = %[[C0]] to %[[C14]] step %[[C1]] iter_args(%[[OUT0:.+]] = %[[OUT_TILE]]) -> (tensor<1x14x14x36xf32>)
//       CHECK:     %[[MLOOP1:.+]] = scf.for %[[M1:.+]] = %[[C0]] to %[[C14]] step %[[C1]] iter_args(%[[OUT1:.+]] = %[[OUT0]]) -> (tensor<1x14x14x36xf32>)
//       CHECK:       %[[KLOOP:.+]] = scf.for %[[K:.+]] = %[[C0]] to %[[C36]] step %[[C1]] iter_args(%[[OUT2:.+]] = %[[OUT1]]) -> (tensor<1x14x14x36xf32>)
//   CHECK-DAG:         %[[kParts:.+]]:3 = affine.delinearize_index %[[K]] into (4, 3, 3) : index, index, index
//   CHECK-DAG:         %[[hIDX:.+]] = affine.apply #[[$MAP]](%[[M0]], %[[kParts]]#1)
//   CHECK-DAG:         %[[wIDX:.+]] = affine.apply #[[$MAP]](%[[M1]], %[[kParts]]#2)
//       CHECK:         %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[hIDX]], %[[wIDX]], %[[kParts]]#0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x16x16x4xf32> to tensor<1x1x1x1xf32>
//       CHECK:         %[[OUT_SLICE:.+]] = tensor.extract_slice %[[OUT2]][0, %[[M0]], %[[M1]], %[[K]]] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x14x14x36xf32> to tensor<1x1x1x1xf32>
//       CHECK:         %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<1x1x1x1xf32>) outs(%[[OUT_SLICE]] : tensor<1x1x1x1xf32>)
//       CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[OUT2]][0, %[[M0]], %[[M1]], %[[K]]] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x14x14x36xf32>
//       CHECK:         scf.yield %[[INSERT]] : tensor<1x14x14x36xf32>
//       CHECK:       scf.yield %[[KLOOP]] : tensor<1x14x14x36xf32>
//       CHECK:     scf.yield %[[MLOOP1]] : tensor<1x14x14x36xf32>
//       CHECK:   return %[[MLOOP0]] : tensor<1x14x14x36xf32>

// CHECK-UNROLL-LABEL: func.func @im2col_nhwc_with_perm
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
//       CHECK-UNROLL:   linalg.copy

// -----

// Test 10: CHWN layout with batch dim as innermost.
// Static sizes; three nested loops (M0, M1, K). Spatial coords via affine.apply.
// The batch dim is innermost so extract_slice reads a 4-element slice (the full batch).
// Because output_perm is [0,1,2,3] but batch_pos = [3], the output dimension order
// differs from the input slice order, so a linalg.transpose is needed.
module {
  func.func @im2col_chwn(%arg0: tensor<16x26x18x4xf32>, %arg1: index, %arg2: index, %arg3: index) -> tensor<4x2x2x2xf32> {
    %0 = tensor.empty() : tensor<4x2x2x2xf32>
    %1 = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [1, 1] kernel_size = [24, 16]
            offsets = [0, %arg1, %arg2, %arg3] output_sizes = [[4], [3], [3], [16, 24, 16]]
            batch_pos = [3] m_pos = [1, 2] k_pos = [0]
            input_k_perm = [0, 1, 2] output_perm = [0, 1, 2, 3]
            ins(%arg0 : tensor<16x26x18x4xf32>)
            outs(%0 : tensor<4x2x2x2xf32>) -> tensor<4x2x2x2xf32>
    return %1 : tensor<4x2x2x2xf32>
  }
}

//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1)[s0] -> (d0 + d1 + s0)>
// CHECK-LABEL: func.func @im2col_chwn
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: tensor<16x26x18x4xf32>
//  CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME: %[[ARG3:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
//       CHECK: %[[INIT:.+]] = tensor.empty() : tensor<4x2x2x2xf32>
//       CHECK: %[[mLOOP0:.+]] = scf.for %[[M0:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUT0:.+]] = %[[INIT]])
//       CHECK:   %[[mLOOP1:.+]] = scf.for %[[M1:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUT1:.+]] = %[[OUT0]])
//       CHECK:     %[[kLOOP:.+]] = scf.for %[[K:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUT2:.+]] = %[[OUT1]])
//   CHECK-DAG:       %[[kIDX:.+]] = affine.apply #[[$MAP]](%[[K]])[%[[ARG3]]]
//   CHECK-DAG:       %[[kParts:.+]]:3 = affine.delinearize_index %[[kIDX]] into (16, 24, 16)
//   CHECK-DAG:       affine.apply #[[$MAP1]](%[[kParts]]#1, %[[M0]])[%[[ARG1]]]
//   CHECK-DAG:       affine.apply #[[$MAP1]](%[[kParts]]#2, %[[M1]])[%[[ARG2]]]
//       CHECK:       %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[kParts]]#0, {{.*}}, {{.*}}, 0] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<16x26x18x4xf32> to tensor<1x1x1x4xf32>
//       CHECK:       %[[DEST_SLICE:.+]] = tensor.extract_slice %[[OUT2]][0, %[[M0]], %[[M1]], %[[K]]] [4, 1, 1, 1] [1, 1, 1, 1] : tensor<4x2x2x2xf32> to tensor<4x1x1x1xf32>
//       CHECK:       %[[TRANS:.+]] = linalg.transpose ins(%[[IN_SLICE]] : tensor<1x1x1x4xf32>) outs(%[[DEST_SLICE]] : tensor<4x1x1x1xf32>) permutation = [3, 1, 2, 0]
//       CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[TRANS]] into %[[OUT2]][0, %[[M0]], %[[M1]], %[[K]]] [4, 1, 1, 1] [1, 1, 1, 1] : tensor<4x1x1x1xf32> into tensor<4x2x2x2xf32>
//       CHECK:      scf.yield %[[INSERT]] : tensor<4x2x2x2xf32>
//       CHECK:    scf.yield %[[kLOOP]] : tensor<4x2x2x2xf32>
//       CHECK:  scf.yield %[[mLOOP1]] : tensor<4x2x2x2xf32>
//       CHECK: return %[[mLOOP0:.+]] : tensor<4x2x2x2xf32>

// CHECK-UNROLL-LABEL: func.func @im2col_chwn
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col

// -----

// Test 11: CHWN layout with output_perm that eliminates the batch transpose.
// Static sizes; three nested loops (K, M0, M1) matching the permuted output.
// Because output_perm = [3, 1, 2, 0], the batch and K positions are swapped
// relative to the default [B, M0, M1, K]. No transpose is needed because
// the output slice is arranged to match the input slice.
module {
  func.func @im2col_chwn_output_perm(%arg0: tensor<16x26x18x4xf32>, %arg1: index, %arg2: index, %arg3: index) -> tensor<2x2x2x4xf32> {
    %0 = tensor.empty() : tensor<2x2x2x4xf32>
    %1 = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [1, 1] kernel_size = [24, 16]
            offsets = [0, %arg1, %arg2, %arg3] output_sizes = [[4], [3], [3], [16, 24, 16]]
            batch_pos = [3] m_pos = [1, 2] k_pos = [0]
            input_k_perm = [0, 1, 2] output_perm = [3, 1, 2, 0]
            ins(%arg0 : tensor<16x26x18x4xf32>)
            outs(%0 : tensor<2x2x2x4xf32>) -> tensor<2x2x2x4xf32>
    return %1 : tensor<2x2x2x4xf32>
  }
}

//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1)[s0] -> (d0 + d1 + s0)>
// CHECK-LABEL: func.func @im2col_chwn_output_perm
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: tensor<16x26x18x4xf32>
//  CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME: %[[ARG3:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
//       CHECK: %[[INIT:.+]] = tensor.empty() : tensor<2x2x2x4xf32>
//       CHECK: %[[LOOP0:.+]] = scf.for %[[IV0:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[INIT]])
//       CHECK:   %[[LOOP1:.+]] = scf.for %[[IV1:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG5:.+]] = %[[ARG4]])
//       CHECK:     %[[LOOP2:.+]] = scf.for %[[IV2:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[ARG5]])
//   CHECK-DAG:       %[[kIDX:.+]] = affine.apply #[[$MAP]](%[[IV0]])[%[[ARG3]]]
//   CHECK-DAG:       %[[kParts:.+]]:3 = affine.delinearize_index %[[kIDX]] into (16, 24, 16)
//   CHECK-DAG:       affine.apply #[[$MAP1]](%[[kParts]]#1, %[[IV1]])[%[[ARG1]]]
//   CHECK-DAG:       affine.apply #[[$MAP1]](%[[kParts]]#2, %[[IV2]])[%[[ARG2]]]
//       CHECK:       %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[kParts]]#0, {{.*}}, {{.*}}, 0] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<16x26x18x4xf32> to tensor<1x1x1x4xf32>
//       CHECK:       %[[OUT_SLICE:.+]] = tensor.extract_slice %[[ARG6]][%[[IV0]], %[[IV1]], %[[IV2]], 0] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<2x2x2x4xf32> to tensor<1x1x1x4xf32>
//       CHECK:       %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<1x1x1x4xf32>) outs(%[[OUT_SLICE]] : tensor<1x1x1x4xf32>)
//       CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[ARG6]][%[[IV0]], %[[IV1]], %[[IV2]], 0] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<1x1x1x4xf32> into tensor<2x2x2x4xf32>
//       CHECK:       scf.yield %[[INSERT]] : tensor<2x2x2x4xf32>
//       CHECK:     scf.yield %[[LOOP2]] : tensor<2x2x2x4xf32>
//       CHECK:   scf.yield %[[LOOP1]] : tensor<2x2x2x4xf32>
//       CHECK: return %[[LOOP0]] : tensor<2x2x2x4xf32>

// CHECK-UNROLL-LABEL: func.func @im2col_chwn_output_perm
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col

// -----

// Test 12: Expanded CHWN layout with output_perm and multi-element batch dims.
// Five nested loops; the batch dims (batch_pos=[3,4]) are innermost in the input
// but appear in loops corresponding to the permuted output positions.
module {
  func.func @im2col_chwn_output_perm_expanded(%arg0: tensor<16x26x18x2x4xf32>, %arg1: index, %arg2: index, %arg3: index) -> tensor<2x2x2x2x2x4xf32> {
    %0 = tensor.empty() : tensor<2x2x2x2x2x4xf32>
    %1 = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [1, 1] kernel_size = [24, 16]
            offsets = [0, 0, %arg1, %arg2, %arg3, 0] output_sizes = [[2], [4], [3], [3], [16, 24], [16]]
            batch_pos = [3, 4] m_pos = [1, 2] k_pos = [0]
            input_k_perm = [0, 1, 2] output_perm = [4, 5, 2, 3, 0, 1]
            ins(%arg0 : tensor<16x26x18x2x4xf32>)
            outs(%0 : tensor<2x2x2x2x2x4xf32>) -> tensor<2x2x2x2x2x4xf32>
    return %1 : tensor<2x2x2x2x2x4xf32>
  }
}

// CHECK-LABEL: func.func @im2col_chwn_output_perm_expanded
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: tensor<16x26x18x2x4xf32>
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
//       CHECK: %[[INIT:.+]] = tensor.empty() : tensor<2x2x2x2x2x4xf32>
//       CHECK: %[[LOOP0:.+]] = scf.for {{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
//       CHECK:   %[[LOOP1:.+]] = scf.for {{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
//       CHECK:     %[[LOOP2:.+]] = scf.for {{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
//       CHECK:       %[[LOOP3:.+]] = scf.for {{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
//       CHECK:         %[[LOOP4:.+]] = scf.for {{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
//       CHECK:           affine.delinearize_index {{.*}} into (16, 24)
//       CHECK:           %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]]{{.*}} [1, 1, 1, 1, 4] [1, 1, 1, 1, 1] : tensor<16x26x18x2x4xf32> to tensor<1x1x1x1x4xf32>
//       CHECK:           linalg.copy ins(%[[IN_SLICE]] : tensor<1x1x1x1x4xf32>)
//       CHECK:           tensor.insert_slice {{.*}} : tensor<1x1x1x1x4xf32> into tensor<2x2x2x2x2x4xf32>
//       CHECK:           scf.yield {{.*}} : tensor<2x2x2x2x2x4xf32>
//       CHECK:         scf.yield %[[LOOP4]] : tensor<2x2x2x2x2x4xf32>
//       CHECK:       scf.yield %[[LOOP3]] : tensor<2x2x2x2x2x4xf32>
//       CHECK:     scf.yield %[[LOOP2]] : tensor<2x2x2x2x2x4xf32>
//       CHECK:   scf.yield %[[LOOP1]] : tensor<2x2x2x2x2x4xf32>
//       CHECK: return %[[LOOP0]] : tensor<2x2x2x2x2x4xf32>

// CHECK-UNROLL-LABEL: func.func @im2col_chwn_output_perm_expanded
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col

// -----

// Test 13: Multiple k_pos entries (k_pos = [0, 2]).
// Verify that both k_pos entries are used for the extract_slice offsets:
// k_pos[0] = input dim 0 (C=16), k_pos[1] = input dim 2 (W=32).
module {
  func.func @im2col_multiple_k_pos(%arg0: tensor<16x52x32x96xbf16>) -> tensor<3x24576x96xbf16> {
    %0 = tensor.empty() : tensor<3x24576x96xbf16>
    %1 = iree_linalg_ext.im2col
            strides = [2] dilations = [1] kernel_size = [48]
            offsets = [0, 0, 0] output_sizes = [[96], [3], [16, 48, 32]]
            batch_pos = [3] m_pos = [1] k_pos = [0, 2]
            input_k_perm = [0, 1, 2] output_perm = [1, 2, 0]
            ins(%arg0 : tensor<16x52x32x96xbf16>)
            outs(%0 : tensor<3x24576x96xbf16>) -> tensor<3x24576x96xbf16>
    return %1 : tensor<3x24576x96xbf16>
  }
}
// CHECK-LABEL: func.func @im2col_multiple_k_pos
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<16x52x32x96xbf16>
//       CHECK:       %[[kParts:.+]]:3 = affine.delinearize_index {{.*}} into (16, 48, 32)
//       CHECK:       tensor.extract_slice %[[ARG0]][%[[kParts]]#0, {{.*}}, %[[kParts]]#2, 0]
//   CHECK-NOT:   iree_linalg_ext.im2col

// CHECK-UNROLL-LABEL: func.func @im2col_multiple_k_pos
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
//       CHECK-UNROLL:   tensor.extract_slice %[[ARG0:.+]][{{.*}}] {{.*}} : tensor<16x52x32x96xbf16>

// -----

// Test 14: CHWN rank-reduced output with dynamic sizes. The batch dim is innermost.
// Nested M and K loops reading a 4-element batch slice.
module {
  func.func @im2col_chwn_rank_reduce(%arg0: tensor<16x26x18x4xf32>, %arg1: index, %arg2: index, %m_size: index, %k_size: index) -> tensor<4x?x?xf32> {
    %0 = tensor.empty(%m_size, %k_size) : tensor<4x?x?xf32>
    %1 = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [1, 1] kernel_size = [24, 16]
            offsets = [0, %arg1, %arg2] output_sizes = [[4], [3, 3], [16, 24, 16]]
            batch_pos = [3] m_pos = [1, 2] k_pos = [0]
            input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
            ins(%arg0 : tensor<16x26x18x4xf32>)
            outs(%0 : tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    return %1 : tensor<4x?x?xf32>
  }
}

// Verify that when the batch dimension is the innermost and generates rank-reduced output,
// a 4-element tensor slice is extracted, copied, and inserted.
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: func.func @im2col_chwn_rank_reduce
//       CHECK:     %[[IN_SLICE:.+]] = tensor.extract_slice {{.*}} : tensor<16x26x18x4xf32> to tensor<4xf32>
//       CHECK:     linalg.copy ins(%[[IN_SLICE]]
//       CHECK:     tensor.insert_slice {{.*}} : tensor<4xf32> into tensor<4x?x?xf32>

// CHECK-UNROLL-LABEL: func.func @im2col_chwn_rank_reduce
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
//       CHECK-UNROLL:   scf.for
//       CHECK-UNROLL:     scf.for
//       CHECK-UNROLL:       linalg.copy

// -----

// Test 15: Backward-weight-style im2col with dilation=2 and expanded M output dims.
// With 2 M output dims each having a single inner size, M coords are used
// directly without delinearization. The spatial offsets include dilation factor.
// Static sizes, non-padded: extract_slice + linalg.copy + insert_slice.
module {
  func.func @im2col_bwd_weight_dilation(%arg0: tensor<4x18x18x2xf32>, %m0: index, %m1: index, %k: index) -> tensor<3x3x4x2xf32> {
    %0 = tensor.empty() : tensor<3x3x4x2xf32>
    %1 = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [2, 2] kernel_size = [8, 8]
            offsets = [0, %m0, %m1, %k] output_sizes = [[2], [3], [3], [4, 8, 8]]
            batch_pos = [3] m_pos = [1, 2] k_pos = [0]
            input_k_perm = [0, 1, 2] output_perm = [1, 2, 3, 0]
            ins(%arg0 : tensor<4x18x18x2xf32>)
            outs(%0 : tensor<3x3x4x2xf32>) -> tensor<3x3x4x2xf32>
    return %1 : tensor<3x3x4x2xf32>
  }
}
// CHECK-LABEL: func.func @im2col_bwd_weight_dilation
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4x18x18x2xf32>
//  CHECK-SAME:     %[[M0:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[M1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[K:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//       CHECK:   %[[INIT:.+]] = tensor.empty() : tensor<3x3x4x2xf32>
//       CHECK:   %[[LOOP0:.+]] = scf.for %[[IV0:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[A0:.+]] = %[[INIT]])
//       CHECK:     %[[LOOP1:.+]] = scf.for %[[IV1:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[A1:.+]] = %[[A0]])
//       CHECK:       %[[LOOP2:.+]] = scf.for %[[IV2:.+]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[A2:.+]] = %[[A1]])
//   CHECK-DAG:         affine.delinearize_index {{.*}} into (4, 8, 8)
//       CHECK:         %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]]{{.*}} [1, 1, 1, 2] [1, 1, 1, 1] : tensor<4x18x18x2xf32> to tensor<1x1x1x2xf32>
//       CHECK:         %[[DEST_SLICE:.+]] = tensor.extract_slice %[[A2]][%[[IV0]], %[[IV1]], %[[IV2]], 0] [1, 1, 1, 2] {{.*}} : tensor<3x3x4x2xf32> to tensor<1x1x1x2xf32>
//       CHECK:         %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<1x1x1x2xf32>) outs(%[[DEST_SLICE]] : tensor<1x1x1x2xf32>)
//       CHECK:         tensor.insert_slice %[[COPY]] into %[[A2]][%[[IV0]], %[[IV1]], %[[IV2]], 0] [1, 1, 1, 2] [1, 1, 1, 1] : tensor<1x1x1x2xf32> into tensor<3x3x4x2xf32>
// Verify the im2col op is fully lowered to loops.
//   CHECK-NOT:   iree_linalg_ext.im2col

// CHECK-UNROLL-LABEL: func.func @im2col_bwd_weight_dilation
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
