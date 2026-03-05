// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-im2col{unroll=false}, canonicalize, cse))" --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-im2col{unroll=true}))" --split-input-file %s | FileCheck %s --check-prefix=CHECK-UNROLL

// Dynamic M offset and K offset -- non-padded: extract_slice + linalg.copy.
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
//   CHECK-DAG: #[[$BMAX:.+]] = affine_map<(d0) -> (0, d0)>
//   CHECK-DAG: #[[$BMIN:.+]] = affine_map<(d0) -> (1, d0)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0)[s0] -> (0, d0 + s0)>
//   CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0) -> (33, d0)>
//   CHECK-DAG: #[[$MAP4:.+]] = affine_map<()[s0] -> (0, s0)>
//   CHECK-DAG: #[[$MAP5:.+]] = affine_map<()[s0] -> (639, s0)>
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
//   CHECK-DAG:       %[[kParts:.+]]:3 = affine.delinearize_index %[[kScaled]] into (3, 640) : index, index, index
//   CHECK-DAG:       %[[mIDX:.+]] = affine.apply #[[$MAP1]](%[[m]])[%[[mOFF]]]
//   CHECK-DAG:       %[[mParts:.+]]:2 = affine.delinearize_index %[[mIDX]] into (32) : index, index
//   CHECK-DAG:       %[[bCLAMP0:.+]] = affine.max #[[$BMAX]](%[[b]])
//   CHECK-DAG:       %[[bCLAMP:.+]] = affine.min #[[$BMIN]](%[[bCLAMP0]])
//   CHECK-DAG:       %[[hCLAMP0:.+]] = affine.max #[[$MAP2]](%[[mParts]]#0)[%[[kParts]]#0]
//   CHECK-DAG:       %[[hCLAMP:.+]] = affine.min #[[$MAP3]](%[[hCLAMP0]])
//   CHECK-DAG:       %[[wCLAMP0:.+]] = affine.max #[[$MAP2]](%[[mParts]]#1)[%[[kParts]]#1]
//   CHECK-DAG:       %[[wCLAMP:.+]] = affine.min #[[$MAP3]](%[[wCLAMP0]])
//   CHECK-DAG:       %[[kCLAMP0:.+]] = affine.max #[[$MAP4]]()[%[[kParts]]#2]
//   CHECK-DAG:       %[[kCLAMP:.+]] = affine.min #[[$MAP5]]()[%[[kCLAMP0]]]
//       CHECK:       %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[bCLAMP]], %[[hCLAMP]], %[[wCLAMP]], %[[kCLAMP]]] [1, 1, 1, 4]
//       CHECK:       linalg.copy ins(%[[IN_SLICE]]
//       CHECK:       tensor.insert_slice {{.*}} into %[[OUT1]][%[[b]], %[[m]], 0] [1, 1, 4] [1, 1, 1] : tensor<4xf32> into tensor<2x?x4xf32>
//       CHECK:       scf.yield {{.*}} : tensor<2x?x4xf32>
//       CHECK:     scf.yield %[[mLOOP]] : tensor<2x?x4xf32>
//       CHECK:   return %[[bLOOP]] : tensor<2x?x4xf32>

// -----

// Dynamic M and K offsets with transposed m_pos -- non-padded: linalg.copy.
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
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (0, d0)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (639, d0)>
//   CHECK-DAG: #[[$BMIN:.+]] = affine_map<(d0) -> (1, d0)>
//   CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0, d1) -> (0, d0 * 3 + d1 * 7)>
//   CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0) -> (100, d0)>
//   CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0, d1) -> (0, d0 * 5 + d1 * 4)>
//   CHECK-DAG: #[[$MAP6:.+]] = affine_map<(d0) -> (171, d0)>
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
//   CHECK-DAG:         %[[kParts:.+]]:3 = affine.delinearize_index %[[kIDX]] into (5, 2) : index, index, index
//   CHECK-DAG:         %[[mIDX:.+]] = affine.apply #[[$MAP]](%[[m]])[%[[mOFF]]]
//   CHECK-DAG:         %[[mParts:.+]]:2 = affine.delinearize_index %[[mIDX]] into (32) : index, index
//   CHECK-DAG:         %[[kCLAMP0:.+]] = affine.max #[[$MAP1]](%[[kParts]]#0)
//   CHECK-DAG:         %[[kCLAMP:.+]] = affine.min #[[$MAP2]](%[[kCLAMP0]])
//   CHECK-DAG:         %[[bCLAMP0:.+]] = affine.max #[[$MAP1]](%[[b]])
//   CHECK-DAG:         %[[bCLAMP:.+]] = affine.min #[[$BMIN]](%[[bCLAMP0]])
//   CHECK-DAG:         %[[wCLAMP0:.+]] = affine.max #[[$MAP3]](%[[mParts]]#1, %[[kParts]]#2)
//   CHECK-DAG:         %[[wCLAMP:.+]] = affine.min #[[$MAP4]](%[[wCLAMP0]])
//   CHECK-DAG:         %[[hCLAMP0:.+]] = affine.max #[[$MAP5]](%[[mParts]]#0, %[[kParts]]#1)
//   CHECK-DAG:         %[[hCLAMP:.+]] = affine.min #[[$MAP6]](%[[hCLAMP0]])
//       CHECK:         %[[IN_SLICE2:.+]] = tensor.extract_slice %[[ARG0]][%[[kCLAMP]], %[[bCLAMP]], %[[wCLAMP]], %[[hCLAMP]]] [1, 1, 1, 1]
//       CHECK:         linalg.copy ins(%[[IN_SLICE2]]
//       CHECK:         tensor.insert_slice {{.*}} into %[[OUT2]][%[[b]], %[[m]], %[[k]]] [1, 1, 1] [1, 1, 1] : tensor<1xf32> into tensor<2x?x?xf32>
//       CHECK:         scf.yield {{.*}} : tensor<2x?x?xf32>
//       CHECK:       scf.yield %[[kLOOP]] : tensor<2x?x?xf32>
//       CHECK:     scf.yield %[[mLOOP]] : tensor<2x?x?xf32>
//       CHECK:   return %[[bLOOP]] : tensor<2x?x?xf32>

// -----

// Static sizes -- non-padded: extract_slice + linalg.copy + insert_slice.
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
//   CHECK-DAG: #[[$BMAX:.+]] = affine_map<(d0) -> (0, d0)>
//   CHECK-DAG: #[[$BMIN:.+]] = affine_map<(d0) -> (1, d0)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1)[s0] -> (0, d0 + d1 + s0)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (33, d0)>
// CHECK-LABEL: func.func @im2col_expanded
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[mSIZE0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[mSIZE1:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[mOFF0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[mOFF1:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[kOFF:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[mSTRIDE:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[OUT_TILE:.+]] = tensor.empty(%[[mSIZE0]], %[[mSIZE1]]) : tensor<2x?x?x2x4xf32>
//       CHECK:   %[[bLOOP:.+]] = scf.for %[[b:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUT0:.+]] = %[[OUT_TILE]]) -> (tensor<2x?x?x2x4xf32>)
//       CHECK:     %[[mLOOP0:.+]] = scf.for %[[m0:.+]] = %[[C0]] to %[[mSIZE0]] step %[[C1]] iter_args(%[[OUT1:.+]] = %[[OUT0]]) -> (tensor<2x?x?x2x4xf32>)
//       CHECK:       %[[mLOOP1:.+]] = scf.for %[[m1:.+]] = %[[C0]] to %[[mSIZE1]] step %[[C1]] iter_args(%[[OUT2:.+]] = %[[OUT1]]) -> (tensor<2x?x?x2x4xf32>)
//       CHECK:         %[[kLOOP:.+]] = scf.for %[[k:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUT3:.+]] = %[[OUT2]]) -> (tensor<2x?x?x2x4xf32>)
//   CHECK-DAG:           %[[kIDX:.+]] = affine.apply #[[$MAP]](%[[k]])[%[[kOFF]]]
//   CHECK-DAG:           %[[kParts:.+]]:2 = affine.delinearize_index %[[kIDX]] into (3) : index, index
//   CHECK-DAG:           %[[bCLAMP0:.+]] = affine.max #[[$BMAX]](%[[b]])
//   CHECK-DAG:           %[[bCLAMP:.+]] = affine.min #[[$BMIN]](%[[bCLAMP0]])
//   CHECK-DAG:           %[[hCLAMP0:.+]] = affine.max #[[$MAP1]](%[[kParts]]#0, %[[m0]])[%[[mOFF0]]]
//   CHECK-DAG:           %[[hCLAMP:.+]] = affine.min #[[$MAP2]](%[[hCLAMP0]])
//   CHECK-DAG:           %[[wCLAMP0:.+]] = affine.max #[[$MAP1]](%[[kParts]]#1, %[[m1]])[%[[mOFF1]]]
//   CHECK-DAG:           %[[wCLAMP:.+]] = affine.min #[[$MAP2]](%[[wCLAMP0]])
//       CHECK:           %[[DEST_SLICE:.+]] = tensor.extract_slice %[[OUT3]][%[[b]], %[[m0]], %[[m1]], %[[k]], 0] [1, 1, 1, 1, 4] {{.*}} : tensor<2x?x?x2x4xf32> to tensor<4xf32>
//       CHECK:           %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[bCLAMP]], %[[hCLAMP]], %[[wCLAMP]], 0] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<2x34x34x640xf32> to tensor<4xf32>
//       CHECK:           %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<4xf32>) outs(%[[DEST_SLICE]] : tensor<4xf32>)
//       CHECK:           %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[OUT3]][%[[b]], %[[m0]], %[[m1]], %[[k]], 0] [1, 1, 1, 1, 4] [1, 1, 1, 1, 1] : tensor<4xf32> into tensor<2x?x?x2x4xf32>
//       CHECK:           scf.yield %[[INSERT]] : tensor<2x?x?x2x4xf32>
//       CHECK:         scf.yield %[[kLOOP]] : tensor<2x?x?x2x4xf32>
//       CHECK:       scf.yield %[[mLOOP1]] : tensor<2x?x?x2x4xf32>
//       CHECK:     scf.yield %[[mLOOP0]] : tensor<2x?x?x2x4xf32>
//       CHECK:   return %[[bLOOP]] : tensor<2x?x?x2x4xf32>

// -----

// NCHW layout with static sizes -- non-padded: extract_slice + linalg.copy + insert_slice.
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
// Scalar fallback (34 % 4 != 0): loops over batch + K dims with linalg.copy.
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (0, d0 + s0)>
// CHECK-LABEL: func.func @im2col_expanded_nchw
//       CHECK:   tensor.extract_slice {{.*}} : tensor<2x640x34x34xf32> to tensor<1xf32>
//       CHECK:   linalg.copy
//       CHECK:   tensor.insert_slice {{.*}} : tensor<1xf32> into tensor<2x1x1x2x4xf32>

// -----

// Test backward-weight-style im2col where M dims are kernel spatial dims,
// not convolution output spatial dims. With 2 expanded M output dims,
// no delinearization is needed -- the M coords are used directly.
// Non-padded: extract_slice + linalg.copy.
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
// With expanded M (2 single-element M output dims), M coords are used directly
// without delinearization. Non-padded: extract_slice + linalg.copy.
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> (0, s0)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (7, s0)>
// CHECK-LABEL: func.func @im2col_bwd_weight_dilation2
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1x8x8xf32>
//  CHECK-SAME: %[[M0:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME: %[[M1:[a-zA-Z0-9_]+]]: index
//       CHECK: %[[M0_LO:.+]] = affine.max #[[$MAP]]()[%[[M0]]]
//       CHECK: %[[M0_CLAMP:.+]] = affine.min #[[$MAP1]]()[%[[M0_LO]]]
//       CHECK: %[[M1_LO:.+]] = affine.max #[[$MAP]]()[%[[M1]]]
//       CHECK: %[[M1_CLAMP:.+]] = affine.min #[[$MAP1]]()[%[[M1_LO]]]
//       CHECK: tensor.extract_slice %[[ARG0]][0, 0, %[[M0_CLAMP]], %[[M1_CLAMP]]] [1, 1, 1, 1]
//       CHECK: linalg.copy
//       CHECK: return

// -----

#map = affine_map<(d0) -> (d0 * 4)>
module {
  func.func @im2col_unrolled(%arg0: tensor<2x34x34x640xf32>, %m_off: index, %k: index) -> tensor<2x2x4xf32> {
    %0 = tensor.empty() : tensor<2x2x4xf32>
    %k_off = affine.apply #map(%k)
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
// CHECK-UNROLL-LABEL: func.func @im2col_unrolled
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
//       CHECK-UNROLL:   %[[EMPTY:.*]] = tensor.empty() : tensor<2x2x4xf32>
//       CHECK-UNROLL:   linalg.copy
//       CHECK-UNROLL:   %[[INS0:.*]] = tensor.insert_slice %{{.*}} into %[[EMPTY]]
//       CHECK-UNROLL:   tensor.extract_slice %[[INS0]]
//       CHECK-UNROLL:   linalg.copy
//       CHECK-UNROLL:   %[[INS1:.*]] = tensor.insert_slice %{{.*}} into %[[INS0]]
//       CHECK-UNROLL:   tensor.extract_slice %[[INS1]]
//       CHECK-UNROLL:   linalg.copy
//       CHECK-UNROLL:   %[[INS2:.*]] = tensor.insert_slice %{{.*}} into %[[INS1]]
//       CHECK-UNROLL:   tensor.extract_slice %[[INS2]]
//       CHECK-UNROLL:   linalg.copy
//       CHECK-UNROLL:   %[[INS3:.*]] = tensor.insert_slice %{{.*}} into %[[INS2]]
//       CHECK-UNROLL:   return %[[INS3]]

// -----

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

// -----

// Static sizes, NHWC layout with input_k_perm -- non-padded: linalg.copy.
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
//       CHECK:       affine.delinearize_index %[[K]] into (2) : index, index
//       CHECK:       tensor.extract_slice %[[ARG0]]
//       CHECK:       linalg.copy
//       CHECK:       tensor.insert_slice {{.*}} into %[[OUT2]][0, %[[M]], %[[K]]] [1, 1, 1] [1, 1, 1] : tensor<1xf32> into tensor<1x2x4xf32>
//       CHECK:       scf.yield {{.*}} : tensor<1x2x4xf32>
//       CHECK:     scf.yield %[[KLOOP]] : tensor<1x2x4xf32>
//       CHECK:   return %[[MLOOP]] : tensor<1x2x4xf32>

// -----

// Static sizes with input_k_perm = [2, 0, 1] -- non-padded: linalg.copy.
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
// Scalar fallback with non-identity k_perm: 3 loops (M0 + M1 + K) with linalg.copy.
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: func.func @im2col_nhwc_with_perm
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x16x16x4xf32>
//   CHECK-DAG: %[[C36:.+]] = arith.constant 36 : index
//   CHECK-DAG: %[[C14:.+]] = arith.constant 14 : index
//   CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//       CHECK: %[[OUT_TILE:.+]] = tensor.empty() : tensor<1x14x14x36xf32>
//       CHECK: %[[MLOOP0:.+]] = scf.for %[[M0:.+]] = %[[C0]] to %[[C14]] step %[[C1]] iter_args(%[[OUT0:.+]] = %[[OUT_TILE]]) -> (tensor<1x14x14x36xf32>)
//       CHECK:   %[[MLOOP1:.+]] = scf.for %[[M1:.+]] = %[[C0]] to %[[C14]] step %[[C1]] iter_args(%[[OUT1:.+]] = %[[OUT0]]) -> (tensor<1x14x14x36xf32>)
//       CHECK:     %[[KLOOP:.+]] = scf.for %[[K:.+]] = %[[C0]] to %[[C36]] step %[[C1]] iter_args(%[[OUT2:.+]] = %[[OUT1]]) -> (tensor<1x14x14x36xf32>)
//       CHECK:       affine.delinearize_index %[[K]] into (3, 3) : index, index, index
//       CHECK:       tensor.extract_slice %[[ARG0]]
//       CHECK:       linalg.copy
//       CHECK:       tensor.insert_slice {{.*}} into %[[OUT2]][0, %[[M0]], %[[M1]], %[[K]]] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1xf32> into tensor<1x14x14x36xf32>
//       CHECK:       scf.yield {{.*}} : tensor<1x14x14x36xf32>
//       CHECK:     scf.yield %[[KLOOP]] : tensor<1x14x14x36xf32>
//       CHECK:   scf.yield %[[MLOOP1]] : tensor<1x14x14x36xf32>
//       CHECK: return %[[MLOOP0]] : tensor<1x14x14x36xf32>

// -----

// CHWN layout with batch dim as innermost.
// Static sizes, non-padded: extract_slice + linalg.copy + insert_slice.
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
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (0, d0)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (15, d0)>
//   CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0, d1)[s0] -> (0, d0 + d1 + s0)>
//   CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0) -> (25, d0)>
//   CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0) -> (17, d0)>
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
//   CHECK-DAG:       %[[kParts:.+]]:3 = affine.delinearize_index %[[kIDX]] into (24, 16) : index, index, index
//   CHECK-DAG:       %[[kCLAMP0:.+]] = affine.max #[[$MAP1]](%[[kParts]]#0)
//   CHECK-DAG:       %[[kCLAMP:.+]] = affine.min #[[$MAP2]](%[[kCLAMP0]])
//   CHECK-DAG:       %[[hCLAMP0:.+]] = affine.max #[[$MAP3]](%[[kParts]]#1, %[[M0]])[%[[ARG1]]]
//   CHECK-DAG:       %[[hCLAMP:.+]] = affine.min #[[$MAP4]](%[[hCLAMP0]])
//   CHECK-DAG:       %[[wCLAMP0:.+]] = affine.max #[[$MAP3]](%[[kParts]]#2, %[[M1]])[%[[ARG2]]]
//   CHECK-DAG:       %[[wCLAMP:.+]] = affine.min #[[$MAP5]](%[[wCLAMP0]])
//       CHECK:       %[[DEST_SLICE:.+]] = tensor.extract_slice %[[OUT2]][0, %[[M0]], %[[M1]], %[[K]]] [4, 1, 1, 1] {{.*}} : tensor<4x2x2x2xf32> to tensor<4xf32>
//       CHECK:       %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[kCLAMP]], %[[hCLAMP]], %[[wCLAMP]], 0] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<16x26x18x4xf32> to tensor<4xf32>
//       CHECK:       %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<4xf32>) outs(%[[DEST_SLICE]] : tensor<4xf32>)
//       CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[OUT2]][0, %[[M0]], %[[M1]], %[[K]]] [4, 1, 1, 1] [1, 1, 1, 1] : tensor<4xf32> into tensor<4x2x2x2xf32>
//       CHECK:      scf.yield %[[INSERT]] : tensor<4x2x2x2xf32>
//       CHECK:    scf.yield %[[kLOOP]] : tensor<4x2x2x2xf32>
//       CHECK:  scf.yield %[[mLOOP1]] : tensor<4x2x2x2xf32>
//       CHECK: return %[[mLOOP0:.+]] : tensor<4x2x2x2xf32>

// -----

// CHWN layout with output_perm that eliminates the transpose.
// Static sizes, non-padded: extract_slice + linalg.copy + insert_slice (no transpose).
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
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (0, d0)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (15, d0)>
//   CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0, d1)[s0] -> (0, d0 + d1 + s0)>
//   CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0) -> (25, d0)>
//   CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0) -> (17, d0)>
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
//   CHECK-DAG:       %[[kParts:.+]]:3 = affine.delinearize_index %[[kIDX]] into (24, 16) : index, index, index
//   CHECK-DAG:       %[[kCLAMP0:.+]] = affine.max #[[$MAP1]](%[[kParts]]#0)
//   CHECK-DAG:       %[[kCLAMP:.+]] = affine.min #[[$MAP2]](%[[kCLAMP0]])
//   CHECK-DAG:       %[[hCLAMP0:.+]] = affine.max #[[$MAP3]](%[[kParts]]#1, %[[IV1]])[%[[ARG1]]]
//   CHECK-DAG:       %[[hCLAMP:.+]] = affine.min #[[$MAP4]](%[[hCLAMP0]])
//   CHECK-DAG:       %[[wCLAMP0:.+]] = affine.max #[[$MAP3]](%[[kParts]]#2, %[[IV2]])[%[[ARG2]]]
//   CHECK-DAG:       %[[wCLAMP:.+]] = affine.min #[[$MAP5]](%[[wCLAMP0]])
//       CHECK:       %[[DEST_SLICE:.+]] = tensor.extract_slice %[[ARG6]][%[[IV0]], %[[IV1]], %[[IV2]], 0] [1, 1, 1, 4] {{.*}} : tensor<2x2x2x4xf32> to tensor<4xf32>
//       CHECK:       %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[kCLAMP]], %[[hCLAMP]], %[[wCLAMP]], 0] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<16x26x18x4xf32> to tensor<4xf32>
//       CHECK:       %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<4xf32>) outs(%[[DEST_SLICE]] : tensor<4xf32>)
//       CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[ARG6]][%[[IV0]], %[[IV1]], %[[IV2]], 0] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<4xf32> into tensor<2x2x2x4xf32>
//       CHECK:       scf.yield %[[INSERT]] : tensor<2x2x2x4xf32>
//       CHECK:     scf.yield %[[LOOP2]] : tensor<2x2x2x4xf32>
//       CHECK:   scf.yield %[[LOOP1]] : tensor<2x2x2x4xf32>
//       CHECK: return %[[LOOP0]] : tensor<2x2x2x4xf32>

// -----

// Expanded CHWN layout with output_perm. Static sizes, non-padded: linalg.copy.
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

//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (0, d0)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (15, d0)>
//   CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0, d1)[s0] -> (0, d0 + d1 + s0)>
//   CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0) -> (25, d0)>
//   CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0) -> (17, d0)>
//   CHECK-DAG: #[[$BMIN:.+]] = affine_map<(d0) -> (1, d0)>
// CHECK-LABEL: func.func @im2col_chwn_output_perm_expanded
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: tensor<16x26x18x2x4xf32>
//  CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME: %[[ARG3:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
//       CHECK: %[[INIT:.+]] = tensor.empty() : tensor<2x2x2x2x2x4xf32>
//       CHECK: %[[LOOP0:.+]] = scf.for %[[IV0:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[INIT]])
//       CHECK:   %[[LOOP1:.+]] = scf.for %[[IV1:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG5:.+]] = %[[ARG4]])
//       CHECK:     %[[LOOP2:.+]] = scf.for %[[IV2:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[ARG5]])
//       CHECK:       %[[LOOP3:.+]] = scf.for %[[IV3:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG7:.+]] = %[[ARG6]])
//       CHECK:         %[[LOOP4:.+]] = scf.for %[[IV4:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.+]] = %[[ARG7]])
//   CHECK-DAG:           %[[kIDX:.+]] = affine.apply #[[$MAP]](%[[IV0]])[%[[ARG3]]]
//   CHECK-DAG:           %[[kParts:.+]]:2 = affine.delinearize_index %[[kIDX]] into (24) : index, index
//   CHECK-DAG:           %[[kCLAMP0:.+]] = affine.max #[[$MAP1]](%[[kParts]]#0)
//   CHECK-DAG:           %[[kCLAMP:.+]] = affine.min #[[$MAP2]](%[[kCLAMP0]])
//   CHECK-DAG:           %[[hCLAMP0:.+]] = affine.max #[[$MAP3]](%[[kParts]]#1, %[[IV2]])[%[[ARG1]]]
//   CHECK-DAG:           %[[hCLAMP:.+]] = affine.min #[[$MAP4]](%[[hCLAMP0]])
//   CHECK-DAG:           %[[wCLAMP0:.+]] = affine.max #[[$MAP3]](%[[IV1]], %[[IV3]])[%[[ARG2]]]
//   CHECK-DAG:           %[[wCLAMP:.+]] = affine.min #[[$MAP5]](%[[wCLAMP0]])
//   CHECK-DAG:           %[[bCLAMP0:.+]] = affine.max #[[$MAP1]](%[[IV4]])
//   CHECK-DAG:           %[[bCLAMP:.+]] = affine.min #[[$BMIN]](%[[bCLAMP0]])
//       CHECK:           %[[DEST_SLICE:.+]] = tensor.extract_slice %[[ARG8]][%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]], %[[IV4]], 0] [1, 1, 1, 1, 1, 4] {{.*}} : tensor<2x2x2x2x2x4xf32> to tensor<4xf32>
//       CHECK:           %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[kCLAMP]], %[[hCLAMP]], %[[wCLAMP]], %[[bCLAMP]], 0] [1, 1, 1, 1, 4] [1, 1, 1, 1, 1] : tensor<16x26x18x2x4xf32> to tensor<4xf32>
//       CHECK:           %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<4xf32>) outs(%[[DEST_SLICE]] : tensor<4xf32>)
//       CHECK:           %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[ARG8]][%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]], %[[IV4]], 0] [1, 1, 1, 1, 1, 4] [1, 1, 1, 1, 1, 1] : tensor<4xf32> into tensor<2x2x2x2x2x4xf32>
//       CHECK:           scf.yield %[[INSERT]] : tensor<2x2x2x2x2x4xf32>
//       CHECK:         scf.yield %[[LOOP4]] : tensor<2x2x2x2x2x4xf32>
//       CHECK:       scf.yield %[[LOOP3]] : tensor<2x2x2x2x2x4xf32>
//       CHECK:     scf.yield %[[LOOP2]] : tensor<2x2x2x2x2x4xf32>
//       CHECK:   scf.yield %[[LOOP1]] : tensor<2x2x2x2x2x4xf32>
//       CHECK: return %[[LOOP0]] : tensor<2x2x2x2x2x4xf32>

// -----

// CHWN rank-reduced output with dynamic sizes. The batch dim is innermost.
// Non-padded: extract_slice + linalg.copy + insert_slice.
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
// a 1D tensor slice is extracted, copied, and inserted (no transpose).
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (0, d0 + d1)>
// CHECK-LABEL: func.func @im2col_chwn_rank_reduce
//       CHECK:     %[[IN_SLICE:.+]] = tensor.extract_slice {{.*}} : tensor<16x26x18x4xf32> to tensor<4xf32>
//       CHECK:     linalg.copy ins(%[[IN_SLICE]]
//       CHECK:     tensor.insert_slice {{.*}} : tensor<4xf32> into tensor<4x?x?xf32>

// -----

// Backward-weight-style im2col with dilation=2 and expanded M output dims.
// With 2 M output dims each having a single inner size, M coords are used
// directly without delinearization. The spatial offsets include dilation factor.
// Static sizes, non-padded: extract_slice + linalg.copy + insert_slice.
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (0, d0)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (3, d0)>
//   CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0, d1)[s0] -> (0, d0 * 2 + d1 + s0)>
//   CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0) -> (17, d0)>
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
//   CHECK-DAG:         %[[kIDX:.+]] = affine.apply #[[$MAP]](%[[IV2]])[%[[K]]]
//   CHECK-DAG:         %[[kParts:.+]]:3 = affine.delinearize_index %[[kIDX]] into (8, 8) : index, index, index
// Verify dilation factor 2 is applied to spatial offset computation.
//   CHECK-DAG:         %[[kCLAMP0:.+]] = affine.max #[[$MAP1]](%[[kParts]]#0)
//   CHECK-DAG:         %[[kCLAMP:.+]] = affine.min #[[$MAP2]](%[[kCLAMP0]])
//   CHECK-DAG:         %[[hCLAMP0:.+]] = affine.max #[[$MAP3]](%[[kParts]]#1, %[[IV0]])[%[[M0]]]
//   CHECK-DAG:         %[[hCLAMP:.+]] = affine.min #[[$MAP4]](%[[hCLAMP0]])
//   CHECK-DAG:         %[[wCLAMP0:.+]] = affine.max #[[$MAP3]](%[[kParts]]#2, %[[IV1]])[%[[M1]]]
//   CHECK-DAG:         %[[wCLAMP:.+]] = affine.min #[[$MAP4]](%[[wCLAMP0]])
//       CHECK:         %[[DEST_SLICE:.+]] = tensor.extract_slice %[[A2]][%[[IV0]], %[[IV1]], %[[IV2]], 0] [1, 1, 1, 2] {{.*}} : tensor<3x3x4x2xf32> to tensor<2xf32>
//       CHECK:         %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[kCLAMP]], %[[hCLAMP]], %[[wCLAMP]], 0] [1, 1, 1, 2] [1, 1, 1, 1] : tensor<4x18x18x2xf32> to tensor<2xf32>
//       CHECK:         %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<2xf32>) outs(%[[DEST_SLICE]] : tensor<2xf32>)
//       CHECK:         tensor.insert_slice %[[COPY]] into %[[A2]][%[[IV0]], %[[IV1]], %[[IV2]], 0] [1, 1, 1, 2] [1, 1, 1, 1] : tensor<2xf32> into tensor<3x3x4x2xf32>
// Verify the im2col op is fully lowered to loops.
//   CHECK-NOT:   iree_linalg_ext.im2col

// -----

// Multi-dim M+K decomposition: both M and K have multi-element inner sizes
// in their output_sizes, requiring delinearize_index for both.
// Non-padded: extract_slice + linalg.copy + insert_slice.
module {
  func.func @im2col_multi_dim_m_and_k(%arg0: tensor<1x20x20x128xf32>, %m_off: index, %k_off: index) -> tensor<1x2x4xf32> {
    %0 = tensor.empty() : tensor<1x2x4xf32>
    %1 = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [1, 1] kernel_size = [5, 5]
            offsets = [0, %m_off, %k_off] output_sizes = [[1], [16, 16], [5, 5, 128]]
            batch_pos = [0] m_pos = [1, 2] k_pos = [3]
            input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
            ins(%arg0 : tensor<1x20x20x128xf32>)
            outs(%0 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
    return %1 : tensor<1x2x4xf32>
  }
}
// CHECK-LABEL: func.func @im2col_multi_dim_m_and_k
// Verify K delinearization into kernel + channel coords.
//       CHECK:   affine.delinearize_index %{{.+}} into (5, 128) : index, index, index
// Verify M delinearization into spatial coords.
//       CHECK:   affine.delinearize_index %{{.+}} into (16) : index, index
// Verify extract_slice + linalg.copy + insert_slice.
//       CHECK:   tensor.extract_slice
//       CHECK:   linalg.copy
//       CHECK:   tensor.insert_slice
// Verify im2col is fully decomposed.
//   CHECK-NOT:   iree_linalg_ext.im2col

// -----

// Padded im2col with scalar fallback: 2 loops (M + K) with padding.
// Verify validity check via affine.max/min factor multiplication,
// K delinearization, pad, and full decomposition.
// CHECK-LABEL: func.func @decompose_padded_im2col
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x5x8xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C24:.+]] = arith.constant 24 : index
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[OUT:.+]] = tensor.empty() : tensor<1x3x24xf32>
//       CHECK:   %[[MLOOP:.+]] = scf.for %[[M:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG2:.+]] = %[[OUT]]) -> (tensor<1x3x24xf32>)
//       CHECK:     %[[KLOOP:.+]] = scf.for %[[K:.+]] = %[[C0]] to %[[C24]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[ARG2]]) -> (tensor<1x3x24xf32>)
//       CHECK:       %{{.+}}:2 = affine.delinearize_index %[[K]] into (8) : index, index
// Verify the validity check uses affine.max/min and arith.muli factor pattern.
//       CHECK:       affine.max
//       CHECK:       affine.max
//       CHECK:       arith.muli
// Verify extract + pad + insert.
//       CHECK:       tensor.extract_slice %[[ARG0]]
//       CHECK:       tensor.pad
//  CHECK-NEXT:       ^bb0
//  CHECK-NEXT:         tensor.yield %[[CST]]
//       CHECK:       tensor.insert_slice {{.*}} into %[[ARG4]][0, %[[M]], %[[K]]] [1, 1, 1] [1, 1, 1]
//   CHECK-NOT:   iree_linalg_ext.im2col
// CHECK-UNROLL-LABEL: func.func @decompose_padded_im2col
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
//       CHECK-UNROLL:   tensor.extract_slice
//       CHECK-UNROLL:   tensor.pad
//       CHECK-UNROLL:   tensor.insert_slice
module {
  func.func @decompose_padded_im2col(%arg0: tensor<1x5x8xf32>) -> tensor<1x3x24xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x3x24xf32>
    %1 = iree_linalg_ext.im2col
        strides = [1] dilations = [1] kernel_size = [3]
        offsets = [0, 0, 0] output_sizes = [[1], [3], [3, 8]]
        batch_pos = [0] m_pos = [1] k_pos = [2]
        input_k_perm = [0, 1] output_perm = [0, 1, 2]
        input_pad_low = [0, 1, 0] input_pad_high = [0, 1, 0]
        pad_value(%cst : f32)
        ins(%arg0 : tensor<1x5x8xf32>) outs(%0 : tensor<1x3x24xf32>) -> tensor<1x3x24xf32>
    return %1 : tensor<1x3x24xf32>
  }
}

// -----

// Padded im2col with vectorized K dim (tile size 8 = channels).
// The K dim is vectorized, batch and M are size 1. After canonicalize+cse,
// the loops are removed and we get straight-line code using affine.min/max
// factor multiplication for bounds checking.
//   CHECK-DAG: #[[$HIGH_MIN:.+]] = affine_map<()[s0] -> (-s0 + 6, 1)>
//   CHECK-DAG: #[[$CLAMP0:.+]] = affine_map<()[s0] -> (0, s0)>
//   CHECK-DAG: #[[$LOW_MIN:.+]] = affine_map<()[s0] -> (1, s0)>
//   CHECK-DAG: #[[$READ_LO:.+]] = affine_map<()[s0] -> (0, s0 - 1)>
//   CHECK-DAG: #[[$READ_HI:.+]] = affine_map<()[s0] -> (4, s0)>
//   CHECK-DAG: #[[$MCHECK:.+]] = affine_map<()[s0] -> (-s0 + 5, 1)>
// CHECK-LABEL: func.func @decompose_padded_im2col_vectorized
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x5x8xf32>
//  CHECK-SAME:     %[[MOFF:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// High bound factor: min(-m_off + 6, 1) clamped to >= 0.
//       CHECK:   %[[HI_MIN:.+]] = affine.min #[[$HIGH_MIN]]()[%[[MOFF]]]
//       CHECK:   %[[HI_OK:.+]] = affine.max #[[$CLAMP0]]()[%[[HI_MIN]]]
// Low bound factor: min(1, m_off) clamped to >= 0.
//       CHECK:   %[[LO_MIN:.+]] = affine.min #[[$LOW_MIN]]()[%[[MOFF]]]
//       CHECK:   %[[LO_OK:.+]] = affine.max #[[$CLAMP0]]()[%[[LO_MIN]]]
// Combined spatial validity factor.
//       CHECK:   %[[FACTOR0:.+]] = arith.muli %[[HI_OK]], %[[LO_OK]] : index
//       CHECK:   %[[VSIZE0:.+]] = arith.muli %[[FACTOR0]], %[[C8]] : index
// Clamped read offset.
//       CHECK:   %[[READ_LO:.+]] = affine.max #[[$READ_LO]]()[%[[MOFF]]]
//       CHECK:   %[[READ_OFF:.+]] = affine.min #[[$READ_HI]]()[%[[READ_LO]]]
// Output M bounds factor: min(-m_off + 5, 1) clamped to >= 0.
//       CHECK:   %[[M_MIN:.+]] = affine.min #[[$MCHECK]]()[%[[MOFF]]]
//       CHECK:   %[[M_OK:.+]] = affine.max #[[$CLAMP0]]()[%[[M_MIN]]]
//       CHECK:   %[[VSIZE:.+]] = arith.muli %[[VSIZE0]], %[[M_OK]] : index
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[READ_OFF]], 0] [1, 1, %[[VSIZE]]] [1, 1, 1]
//       CHECK:   %[[PAD_AMT:.+]] = arith.subi %[[C8]], %[[VSIZE]] : index
//       CHECK:   %[[PADDED:.+]] = tensor.pad %[[SLICE]] low[%[[C0]]] high[%[[PAD_AMT]]]
//       CHECK:     tensor.yield %[[CST]]
//       CHECK:   return {{.*}} : tensor<1x1x8xf32>
// CHECK-UNROLL-LABEL: func.func @decompose_padded_im2col_vectorized
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
//       CHECK-UNROLL:   tensor.extract_slice
//       CHECK-UNROLL:   tensor.pad
module {
  func.func @decompose_padded_im2col_vectorized(%arg0: tensor<1x5x8xf32>, %m_off: index) -> tensor<1x1x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x8xf32>
    %1 = iree_linalg_ext.im2col
        strides = [1] dilations = [1] kernel_size = [3]
        offsets = [0, %m_off, 0] output_sizes = [[1], [5], [3, 8]]
        batch_pos = [0] m_pos = [1] k_pos = [2]
        input_k_perm = [0, 1] output_perm = [0, 1, 2]
        input_pad_low = [0, 1, 0] input_pad_high = [0, 1, 0]
        pad_value(%cst : f32)
        ins(%arg0 : tensor<1x5x8xf32>) outs(%0 : tensor<1x1x8xf32>) -> tensor<1x1x8xf32>
    return %1 : tensor<1x1x8xf32>
  }
}

// -----

// Non-self-inverse input_k_perm = [2, 0, 1]. The inverse permutation
// [1, 2, 0] maps output K coords (ch, kH, kW) -> input order (kH, kW, ch).
// Static sizes, non-padded: extract_slice + linalg.copy + insert_slice.
// Scalar fallback with non-self-inverse k_perm: 3 loops (M0 + M1 + K) with linalg.copy.
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: func.func @im2col_non_self_inverse_kperm
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x16x16x4xf32>
//   CHECK-DAG: %[[C36:.+]] = arith.constant 36 : index
//   CHECK-DAG: %[[C14:.+]] = arith.constant 14 : index
//   CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//       CHECK: %[[OUT:.+]] = tensor.empty() : tensor<1x14x14x36xf32>
//       CHECK: %[[LOOP0:.+]] = scf.for %[[M0:.+]] = %[[C0]] to %[[C14]] step %[[C1]] iter_args(%[[OUT0:.+]] = %[[OUT]])
//       CHECK:   %[[LOOP1:.+]] = scf.for %[[M1:.+]] = %[[C0]] to %[[C14]] step %[[C1]] iter_args(%[[OUT1:.+]] = %[[OUT0]])
//       CHECK:     %[[LOOP2:.+]] = scf.for %[[K:.+]] = %[[C0]] to %[[C36]] step %[[C1]] iter_args(%[[OUT2:.+]] = %[[OUT1]])
//       CHECK:       affine.delinearize_index %[[K]] into (3, 3) : index, index, index
//       CHECK:       tensor.extract_slice %[[ARG0]]
//       CHECK:       linalg.copy
//       CHECK:       tensor.insert_slice {{.*}} into %[[OUT2]][0, %[[M0]], %[[M1]], %[[K]]] [1, 1, 1, 1]
//       CHECK:       scf.yield {{.*}} : tensor<1x14x14x36xf32>
//       CHECK:     scf.yield %[[LOOP2]] : tensor<1x14x14x36xf32>
//       CHECK:   scf.yield %[[LOOP1]] : tensor<1x14x14x36xf32>
//       CHECK: return %[[LOOP0]] : tensor<1x14x14x36xf32>
//   CHECK-NOT: iree_linalg_ext.im2col
module {
  func.func @im2col_non_self_inverse_kperm(%arg0: tensor<1x16x16x4xf32>) -> tensor<1x14x14x36xf32> {
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

// -----

// Padded im2col with non-unit dilation (dilation=2). Verifies that the
// interaction between padding and dilation produces correct offset computation.
// The spatial offset formula is: m_coord * stride + window * dilation - padLow.
// With stride=1, dilation=2, padLow=1, K offset=8 gives window=1 (8/8=1),
// so offset = m_off + 1*2 - 1 = m_off + 1. This distinguishes dilation=2
// from dilation=1, which would give m_off + 1*1 - 1 = m_off.
// The dilation effect shows up in the affine maps:
//   #map2 = (s0) -> (1, s0 + 2) vs (1, s0) for dilation=1.
//   CHECK-DAG: #[[$HIGH_MIN:.+]] = affine_map<()[s0] -> (-s0 + 4, 1)>
//   CHECK-DAG: #[[$CLAMP0:.+]] = affine_map<()[s0] -> (0, s0)>
//   CHECK-DAG: #[[$LOW_MIN:.+]] = affine_map<()[s0] -> (1, s0 + 2)>
//   CHECK-DAG: #[[$READ_LO:.+]] = affine_map<()[s0] -> (0, s0 + 1)>
//   CHECK-DAG: #[[$READ_HI:.+]] = affine_map<()[s0] -> (4, s0)>
//   CHECK-DAG: #[[$MCHECK:.+]] = affine_map<()[s0] -> (-s0 + 3, 1)>
// CHECK-LABEL: func.func @decompose_padded_im2col_dilation
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x5x8xf32>
//  CHECK-SAME:     %[[MOFF:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// High bound factor.
//       CHECK:   %[[HI_MIN:.+]] = affine.min #[[$HIGH_MIN]]()[%[[MOFF]]]
//       CHECK:   %[[HI_OK:.+]] = affine.max #[[$CLAMP0]]()[%[[HI_MIN]]]
// Low bound factor (note +2 from dilation=2).
//       CHECK:   %[[LO_MIN:.+]] = affine.min #[[$LOW_MIN]]()[%[[MOFF]]]
//       CHECK:   %[[LO_OK:.+]] = affine.max #[[$CLAMP0]]()[%[[LO_MIN]]]
// Combined spatial validity factor.
//       CHECK:   %[[FACTOR0:.+]] = arith.muli %[[HI_OK]], %[[LO_OK]] : index
//       CHECK:   %[[VSIZE0:.+]] = arith.muli %[[FACTOR0]], %[[C8]] : index
// Clamped read offset (note +1 from dilation offset).
//       CHECK:   %[[READ_LO:.+]] = affine.max #[[$READ_LO]]()[%[[MOFF]]]
//       CHECK:   %[[READ_OFF:.+]] = affine.min #[[$READ_HI]]()[%[[READ_LO]]]
// Output M bounds factor.
//       CHECK:   %[[M_MIN:.+]] = affine.min #[[$MCHECK]]()[%[[MOFF]]]
//       CHECK:   %[[M_OK:.+]] = affine.max #[[$CLAMP0]]()[%[[M_MIN]]]
//       CHECK:   %[[VSIZE:.+]] = arith.muli %[[VSIZE0]], %[[M_OK]] : index
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[READ_OFF]], 0] [1, 1, %[[VSIZE]]] [1, 1, 1]
//       CHECK:   %[[PAD_AMT:.+]] = arith.subi %[[C8]], %[[VSIZE]] : index
//       CHECK:   %[[PADDED:.+]] = tensor.pad %[[SLICE]] low[%[[C0]]] high[%[[PAD_AMT]]]
//       CHECK:     tensor.yield %[[CST]]
//       CHECK:   return {{.*}} : tensor<1x1x8xf32>
// CHECK-UNROLL-LABEL: func.func @decompose_padded_im2col_dilation
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
//       CHECK-UNROLL:   tensor.extract_slice
//       CHECK-UNROLL:   tensor.pad
module {
  func.func @decompose_padded_im2col_dilation(%arg0: tensor<1x5x8xf32>, %m_off: index) -> tensor<1x1x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x8xf32>
    %1 = iree_linalg_ext.im2col
        strides = [1] dilations = [2] kernel_size = [3]
        offsets = [0, %m_off, 8] output_sizes = [[1], [3], [3, 8]]
        batch_pos = [0] m_pos = [1] k_pos = [2]
        input_k_perm = [0, 1] output_perm = [0, 1, 2]
        input_pad_low = [0, 1, 0] input_pad_high = [0, 1, 0]
        pad_value(%cst : f32)
        ins(%arg0 : tensor<1x5x8xf32>) outs(%0 : tensor<1x1x8xf32>) -> tensor<1x1x8xf32>
    return %1 : tensor<1x1x8xf32>
  }
}

// -----

// 2D spatial padding test: m_pos = [1, 2] with padding on both spatial dims.
// Verifies that validity factors are generated for both spatial dimensions
// and multiplied together, plus output M bounds factors.
//   CHECK-DAG: #[[$HIGH_MIN:.+]] = affine_map<()[s0] -> (-s0 + 6, 1)>
//   CHECK-DAG: #[[$CLAMP0:.+]] = affine_map<()[s0] -> (0, s0)>
//   CHECK-DAG: #[[$LOW_MIN:.+]] = affine_map<()[s0] -> (1, s0)>
//   CHECK-DAG: #[[$READ_LO:.+]] = affine_map<()[s0] -> (0, s0 - 1)>
//   CHECK-DAG: #[[$READ_HI:.+]] = affine_map<()[s0] -> (4, s0)>
//   CHECK-DAG: #[[$MCHECK:.+]] = affine_map<()[s0] -> (-s0 + 7, 1)>
// CHECK-LABEL: func.func @decompose_padded_im2col_2d_spatial
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x5x5x8xf32>
//  CHECK-SAME:     %[[M0:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[M1:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// First spatial dim factor computation.
//       CHECK:   affine.min #[[$HIGH_MIN]]()[%[[M0]]]
//       CHECK:   affine.max #[[$CLAMP0]]()
//       CHECK:   affine.min #[[$LOW_MIN]]()[%[[M0]]]
//       CHECK:   affine.max #[[$CLAMP0]]()
//       CHECK:   arith.muli
//       CHECK:   arith.muli {{.*}}, %[[C8]] : index
// Second spatial dim factor computation.
//       CHECK:   affine.min #[[$HIGH_MIN]]()[%[[M1]]]
//       CHECK:   affine.max #[[$CLAMP0]]()
//       CHECK:   affine.min #[[$LOW_MIN]]()[%[[M1]]]
//       CHECK:   affine.max #[[$CLAMP0]]()
//       CHECK:   arith.muli
//       CHECK:   arith.muli
// Clamped offsets for both dims.
//       CHECK:   %[[CLAMP0_LO:.+]] = affine.max #[[$READ_LO]]()[%[[M0]]]
//       CHECK:   %[[CLAMP0:.+]] = affine.min #[[$READ_HI]]()[%[[CLAMP0_LO]]]
//       CHECK:   %[[CLAMP1_LO:.+]] = affine.max #[[$READ_LO]]()[%[[M1]]]
//       CHECK:   %[[CLAMP1:.+]] = affine.min #[[$READ_HI]]()[%[[CLAMP1_LO]]]
// Output M bounds factors: min(-m + 7, 1) clamped to >= 0.
//       CHECK:   affine.min #[[$MCHECK]]()[%[[M0]]]
//       CHECK:   affine.max #[[$CLAMP0]]()
//       CHECK:   arith.muli
//       CHECK:   affine.min #[[$MCHECK]]()[%[[M1]]]
//       CHECK:   affine.max #[[$CLAMP0]]()
//       CHECK:   %[[VSIZE:.+]] = arith.muli
// Verify extract + pad.
//       CHECK:   tensor.extract_slice %[[ARG0]][0, %[[CLAMP0]], %[[CLAMP1]], 0] [1, 1, 1, %[[VSIZE]]]
//       CHECK:   tensor.pad
//  CHECK-NEXT:   ^bb0
//  CHECK-NEXT:     tensor.yield %[[CST]]
//   CHECK-NOT:   iree_linalg_ext.im2col
module {
  func.func @decompose_padded_im2col_2d_spatial(%arg0: tensor<1x5x5x8xf32>, %m0: index, %m1: index) -> tensor<1x1x1x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x1x8xf32>
    %1 = iree_linalg_ext.im2col
        strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
        offsets = [0, %m0, %m1, 0] output_sizes = [[1], [7], [7], [3, 3, 8]]
        batch_pos = [0] m_pos = [1, 2] k_pos = [3]
        input_k_perm = [0, 1, 2] output_perm = [0, 1, 2, 3]
        input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0]
        pad_value(%cst : f32)
        ins(%arg0 : tensor<1x5x5x8xf32>) outs(%0 : tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xf32>
    return %1 : tensor<1x1x1x8xf32>
  }
}

// -----

// Padding on the vectorized (innermost) input dimension -- decomposition now
// succeeds (Point 7). The vec dim padding generates low and high pad amounts
// using the affine.min/max factor multiplication pattern.
//   CHECK-DAG: #[[$HIGH_MIN:.+]] = affine_map<()[s0] -> (-s0 + 7, 1)>
//   CHECK-DAG: #[[$CLAMP0:.+]] = affine_map<()[s0] -> (0, s0)>
//   CHECK-DAG: #[[$LOW_MIN:.+]] = affine_map<()[s0] -> (1, s0)>
//   CHECK-DAG: #[[$READ_LO:.+]] = affine_map<()[s0] -> (0, s0 - 1)>
//   CHECK-DAG: #[[$READ_HI:.+]] = affine_map<()[s0] -> (5, s0)>
//   CHECK-DAG: #[[$MCHECK:.+]] = affine_map<()[s0] -> (-s0 + 4, 1)>
// CHECK-LABEL: func.func @decompose_padded_on_vec_dim
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x6x8xf32>
//  CHECK-SAME:     %[[MOFF:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[C7:.+]] = arith.constant 7 : index
//   CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// High bound factor.
//       CHECK:   %[[HI_MIN:.+]] = affine.min #[[$HIGH_MIN]]()[%[[MOFF]]]
//       CHECK:   %[[HI_OK:.+]] = affine.max #[[$CLAMP0]]()[%[[HI_MIN]]]
// Low bound factor.
//       CHECK:   %[[LO_MIN:.+]] = affine.min #[[$LOW_MIN]]()[%[[MOFF]]]
//       CHECK:   %[[LO_OK:.+]] = affine.max #[[$CLAMP0]]()[%[[LO_MIN]]]
// Combined factor times valid vec size (7 = spatial valid + vec pad).
//       CHECK:   %[[FACTOR0:.+]] = arith.muli %[[HI_OK]], %[[LO_OK]] : index
//       CHECK:   %[[VSIZE0:.+]] = arith.muli %[[FACTOR0]], %[[C7]] : index
// Clamped read offset.
//       CHECK:   %[[READ_LO:.+]] = affine.max #[[$READ_LO]]()[%[[MOFF]]]
//       CHECK:   %[[READ_OFF:.+]] = affine.min #[[$READ_HI]]()[%[[READ_LO]]]
// Output M bounds factor.
//       CHECK:   %[[M_MIN:.+]] = affine.min #[[$MCHECK]]()[%[[MOFF]]]
//       CHECK:   %[[M_OK:.+]] = affine.max #[[$CLAMP0]]()[%[[M_MIN]]]
//       CHECK:   %[[VSIZE:.+]] = arith.muli %[[VSIZE0]], %[[M_OK]] : index
// Low pad amount = factor * outputMFactor.
//       CHECK:   %[[LOW_PAD:.+]] = arith.muli %[[FACTOR0]], %[[M_OK]] : index
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[READ_OFF]], 0] [1, 1, %[[VSIZE]]] [1, 1, 1]
//       CHECK:   tensor.pad %[[SLICE]] low[%[[LOW_PAD]]] high[%{{.+}}]
//  CHECK-NEXT:   ^bb0
//  CHECK-NEXT:     tensor.yield %[[CST]]
//       CHECK:   return {{.*}} : tensor<1x1x8xf32>
//   CHECK-NOT:   iree_linalg_ext.im2col
// CHECK-UNROLL-LABEL: func.func @decompose_padded_on_vec_dim
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
//       CHECK-UNROLL:   tensor.extract_slice
//       CHECK-UNROLL:   tensor.pad
module {
  func.func @decompose_padded_on_vec_dim(
      %arg0: tensor<1x6x8xf32>,
      %m_off: index) -> tensor<1x1x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x8xf32>
    %1 = iree_linalg_ext.im2col
        strides = [1] dilations = [1] kernel_size = [3]
        offsets = [0, %m_off, 0] output_sizes = [[1], [4], [3, 8]]
        batch_pos = [0] m_pos = [1] k_pos = [2]
        input_k_perm = [0, 1] output_perm = [0, 1, 2]
        input_pad_low = [0, 1, 1] input_pad_high = [0, 1, 1]
        pad_value(%cst : f32)
        ins(%arg0 : tensor<1x6x8xf32>)
        outs(%0 : tensor<1x1x8xf32>) -> tensor<1x1x8xf32>
    return %1 : tensor<1x1x8xf32>
  }
}

// -----

// Output-side M-dimension padding: output M dim tile > product(M output_sizes).
// Input: 1x5x8 (batch=1, spatial=5, channels=8). Kernel [3], stride [1] -> OH=3.
// output_sizes M = [3], product = 3. The tile is 1 in M, so with dynamic offset
// positions beyond M=2 should be all-padding (validSize=0).
// K is vectorized (tile size 8 = channels). Uses factor multiplication pattern.
//   CHECK-DAG: #[[$HIGH_MIN:.+]] = affine_map<()[s0] -> (-s0 + 5, 1)>
//   CHECK-DAG: #[[$CLAMP0:.+]] = affine_map<()[s0] -> (0, s0)>
//   CHECK-DAG: #[[$LOW_MIN:.+]] = affine_map<()[s0] -> (1, s0 + 1)>
//   CHECK-DAG: #[[$MCHECK:.+]] = affine_map<()[s0] -> (-s0 + 3, 1)>
// CHECK-LABEL: func.func @decompose_output_pad_m
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x5x8xf32>
//  CHECK-SAME:     %[[MOFF:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// Input-side bounds check using factor multiplication.
//       CHECK:   affine.min #[[$HIGH_MIN]]()[%[[MOFF]]]
//       CHECK:   affine.max #[[$CLAMP0]]()
//       CHECK:   affine.min #[[$LOW_MIN]]()[%[[MOFF]]]
//       CHECK:   affine.max #[[$CLAMP0]]()
//       CHECK:   arith.muli
//       CHECK:   arith.muli {{.*}}, %[[C8]] : index
// Output-side bounds factor on M dim: min(-m_off + 3, 1) clamped to >= 0.
//       CHECK:   affine.min #[[$MCHECK]]()[%[[MOFF]]]
//       CHECK:   affine.max #[[$CLAMP0]]()
//       CHECK:   arith.muli
//       CHECK:   tensor.extract_slice %[[ARG0]]
//       CHECK:   tensor.pad
//  CHECK-NEXT:   ^bb0
//  CHECK-NEXT:     tensor.yield %[[CST]]
// CHECK-UNROLL-LABEL: func.func @decompose_output_pad_m
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
//       CHECK-UNROLL:   tensor.extract_slice
//       CHECK-UNROLL:   tensor.pad
module {
  func.func @decompose_output_pad_m(%arg0: tensor<1x5x8xf32>, %m_off: index) -> tensor<1x1x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x8xf32>
    %1 = iree_linalg_ext.im2col
        strides = [1] dilations = [1] kernel_size = [3]
        offsets = [0, %m_off, 0] output_sizes = [[1], [3], [3, 8]]
        batch_pos = [0] m_pos = [1] k_pos = [2]
        input_k_perm = [0, 1] output_perm = [0, 1, 2]
        pad_value(%cst : f32)
        ins(%arg0 : tensor<1x5x8xf32>) outs(%0 : tensor<1x1x8xf32>) -> tensor<1x1x8xf32>
    return %1 : tensor<1x1x8xf32>
  }
}

// -----

// Output-side M-dimension padding with combined input padding.
// Input: 1x5x8 with input_pad_low=[0,1,0], input_pad_high=[0,1,0].
// Effective: 1x7x8. Kernel [3], stride [1] -> OH=5.
// output_sizes M = [5], product = 5. Uses factor multiplication pattern
// for both input-side and output-side bounds.
//   CHECK-DAG: #[[$HIGH_MIN:.+]] = affine_map<()[s0] -> (-s0 + 6, 1)>
//   CHECK-DAG: #[[$CLAMP0:.+]] = affine_map<()[s0] -> (0, s0)>
//   CHECK-DAG: #[[$MCHECK:.+]] = affine_map<()[s0] -> (-s0 + 5, 1)>
// CHECK-LABEL: func.func @decompose_output_pad_combined
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x5x8xf32>
//  CHECK-SAME:     %[[MOFF:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// Input-side bounds factors.
//       CHECK:   affine.min #[[$HIGH_MIN]]()[%[[MOFF]]]
//       CHECK:   affine.max #[[$CLAMP0]]()
//       CHECK:   affine.min
//       CHECK:   affine.max #[[$CLAMP0]]()
//       CHECK:   arith.muli
// Output-side bounds factor: min(-m_off + 5, 1) clamped to >= 0.
//       CHECK:   affine.min #[[$MCHECK]]()[%[[MOFF]]]
//       CHECK:   affine.max #[[$CLAMP0]]()
//       CHECK:   arith.muli
//       CHECK:   tensor.extract_slice %[[ARG0]]
//       CHECK:   tensor.pad
//  CHECK-NEXT:   ^bb0
//  CHECK-NEXT:     tensor.yield %[[CST]]
// CHECK-UNROLL-LABEL: func.func @decompose_output_pad_combined
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
//       CHECK-UNROLL:   tensor.extract_slice
//       CHECK-UNROLL:   tensor.pad
module {
  func.func @decompose_output_pad_combined(%arg0: tensor<1x5x8xf32>, %m_off: index) -> tensor<1x1x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x8xf32>
    %1 = iree_linalg_ext.im2col
        strides = [1] dilations = [1] kernel_size = [3]
        offsets = [0, %m_off, 0] output_sizes = [[1], [5], [3, 8]]
        batch_pos = [0] m_pos = [1] k_pos = [2]
        input_k_perm = [0, 1] output_perm = [0, 1, 2]
        input_pad_low = [0, 1, 0] input_pad_high = [0, 1, 0]
        pad_value(%cst : f32)
        ins(%arg0 : tensor<1x5x8xf32>) outs(%0 : tensor<1x1x8xf32>) -> tensor<1x1x8xf32>
    return %1 : tensor<1x1x8xf32>
  }
}

// -----

// Batch > 1 with input padding: verifies that the batch dimension is correctly
// handled in the padded decomposition path. With offset-based batch indexing,
// the batch IV is clamped (affine.max/min) before use in extract_slice.
// CHECK-LABEL: func.func @im2col_batched_with_padding
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//       CHECK:   scf.for %[[BATCH:.*]] = %{{.*}} to %{{.*}}
//       CHECK:     scf.for
//       CHECK:       affine.max {{.*}}(%[[BATCH]])
//       CHECK:       tensor.extract_slice %[[INPUT]]
//       CHECK:       tensor.pad
//       CHECK:       tensor.insert_slice
// CHECK-UNROLL-LABEL: func.func @im2col_batched_with_padding
//   CHECK-UNROLL-NOT:   iree_linalg_ext.im2col
//       CHECK-UNROLL:   tensor.extract_slice
//       CHECK-UNROLL:   tensor.pad
module {
  func.func @im2col_batched_with_padding(%arg0: tensor<2x4x4x3xf32>) -> tensor<2x1x1x27xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x1x1x27xf32>
    %1 = iree_linalg_ext.im2col
        strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
        offsets = [0, 0, 0, 0] output_sizes = [[2], [4], [4], [3, 3, 3]]
        batch_pos = [0] m_pos = [1, 2] k_pos = [3]
        input_k_perm = [0, 1, 2] output_perm = [0, 1, 2, 3]
        input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0]
        pad_value(%cst : f32)
        ins(%arg0 : tensor<2x4x4x3xf32>) outs(%0 : tensor<2x1x1x27xf32>) -> tensor<2x1x1x27xf32>
    return %1 : tensor<2x1x1x27xf32>
  }
}
