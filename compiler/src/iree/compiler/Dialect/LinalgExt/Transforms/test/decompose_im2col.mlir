// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-im2col{unroll=false}, canonicalize, cse))" --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-im2col{unroll=true}))" --split-input-file %s | FileCheck %s --check-prefix=CHECK-UNROLL

#map = affine_map<(d0) -> (d0 * 4)>
module {
  func.func @im2col_untile_k(%arg0: tensor<2x34x34x640xf32>, %m_size: index, %m_off: index, %k: index) -> tensor<2x?x4xf32> {
    %0 = tensor.empty(%m_size) : tensor<2x?x4xf32>
    %k_off = affine.apply #map(%k)
    %7 = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
            m_offset = [%m_off] * [1] k_offset = [%k_off] * [1]
            batch_pos = [0] m_pos = [1, 2] k_pos = [3]
            input_k_perm = [0, 1, 2]
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
//       CHECK:   %[[kScaled:.+]] = affine.apply #[[$MAP]]()[%[[K]]]
//       CHECK:   %[[bLOOP:.+]] = scf.for %[[b:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUT0:.+]] = %[[OUT_TILE]]) -> (tensor<2x?x4xf32>)
//       CHECK:     %[[mLOOP:.+]] = scf.for %[[m:.+]] = %[[C0]] to %[[mSIZE]] step %[[C1]] iter_args(%[[OUT1:.+]] = %[[OUT0]]) -> (tensor<2x?x4xf32>)
//   CHECK-DAG:       %[[kParts:.+]]:3 = affine.delinearize_index %[[kScaled]] into (3, 3, 640)
//   CHECK-DAG:       %[[mIDX:.+]] = affine.apply #[[$MAP1]](%[[m]])[%[[mOFF]]]
//   CHECK-DAG:       %[[mParts:.+]]:2 = affine.delinearize_index %[[mIDX]] into (32, 32)
//   CHECK-DAG:       %[[hIDX:.+]] = affine.apply #[[$MAP1]](%[[mParts]]#0)[%[[kParts]]#0]
//   CHECK-DAG:       %[[wIDX:.+]] = affine.apply #[[$MAP1]](%[[mParts]]#1)[%[[kParts]]#1]
//       CHECK:       %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[b]], %[[hIDX]], %[[wIDX]], %[[kParts]]#2] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<2x34x34x640xf32> to tensor<1x1x4xf32>
//       CHECK:       %[[OUT_SLICE:.+]] = tensor.extract_slice %[[OUT1]][%[[b]], %[[m]], 0] [1, 1, 4] [1, 1, 1] : tensor<2x?x4xf32> to tensor<1x1x4xf32>
//       CHECK:       %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<1x1x4xf32>) outs(%[[OUT_SLICE]] : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
//       CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[OUT1]][%[[b]], %[[m]], 0] [1, 1, 4] [1, 1, 1] : tensor<1x1x4xf32> into tensor<2x?x4xf32>
//       CHECK:       scf.yield %[[INSERT]] : tensor<2x?x4xf32>
//       CHECK:     scf.yield %[[mLOOP]] : tensor<2x?x4xf32>
//       CHECK:   return %[[bLOOP]] : tensor<2x?x4xf32>

// -----

module {
  func.func @im2col_transposed_m_pos(%arg0: tensor<640x2x101x172xf32>, %m_size: index, %k_size: index, %m_off: index, %k_off: index) -> tensor<2x?x?xf32> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty(%m_size, %k_size) : tensor<2x?x?xf32>
    %8 = iree_linalg_ext.im2col
            strides = [5, 3] dilations = [4, 7] kernel_size = [5, 2]
            m_offset = [%m_off] * [1] k_offset = [%k_off] * [1]
            batch_pos = [1] m_pos = [3, 2] k_pos = [0]
            input_k_perm = [0, 1, 2]
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
//   CHECK-DAG:         %[[kParts:.+]]:3 = affine.delinearize_index %[[kIDX]] into (640, 2, 5)
//   CHECK-DAG:         %[[mIDX:.+]] = affine.apply #[[$MAP]](%[[m]])[%[[mOFF]]]
//   CHECK-DAG:         %[[mParts:.+]]:2 = affine.delinearize_index %[[mIDX]] into (32, 32)
//   CHECK-DAG:         %[[hIDX:.+]] = affine.apply #[[$MAP1]](%[[mParts]]#0, %[[kParts]]#1)
//   CHECK-DAG:         %[[wIDX:.+]] = affine.apply #[[$MAP2]](%[[mParts]]#1, %[[kParts]]#2)
//       CHECK:         %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[kParts]]#0, %[[b]], %[[wIDX]], %[[hIDX]]] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<640x2x101x172xf32> to tensor<1x1x1xf32>
//       CHECK:         %[[OUT_SLICE:.+]] = tensor.extract_slice %[[OUT2]][%[[b]], %[[m]], %[[k]]] [1, 1, 1] [1, 1, 1] : tensor<2x?x?xf32> to tensor<1x1x1xf32>
//       CHECK:         %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<1x1x1xf32>) outs(%[[OUT_SLICE]] : tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
//       CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[OUT2]][%[[b]], %[[m]], %[[k]]] [1, 1, 1] [1, 1, 1] : tensor<1x1x1xf32> into tensor<2x?x?xf32>
//       CHECK:         scf.yield %[[INSERT]] : tensor<2x?x?xf32>
//       CHECK:       scf.yield %[[kLOOP]] : tensor<2x?x?xf32>
//       CHECK:     scf.yield %[[mLOOP]] : tensor<2x?x?xf32>
//       CHECK:   return %[[bLOOP]] : tensor<2x?x?xf32>

// -----

module {
  func.func @im2col_expanded(%arg0: tensor<2x34x34x640xf32>, %m_size0: index, %m_size1: index, %m0: index, %m1: index, %k: index, %m_stride: index) -> tensor<2x?x?x2x4xf32> {
    %0 = tensor.empty(%m_size0, %m_size1) : tensor<2x?x?x2x4xf32>
    %7 = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
            m_offset = [%m0, %m1] * [%m_stride, 1] k_offset = [%k, 0] * [4, 1]
            batch_pos = [0] m_pos = [1, 2] k_pos = [3]
            input_k_perm = [0, 1, 2]
            ins(%arg0 : tensor<2x34x34x640xf32>)
            outs(%0 : tensor<2x?x?x2x4xf32>) -> tensor<2x?x?x2x4xf32>
    return %7 : tensor<2x?x?x2x4xf32>
  }
}
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (d0 * 4 + s0 * 4)
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s0 + s0 * s1 + d1 + s2)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
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
//   CHECK-DAG:           %[[kParts:.+]]:3 = affine.delinearize_index %[[kIDX]] into (3, 3, 640)
//   CHECK-DAG:           %[[mIDX:.+]] = affine.apply #[[$MAP1]](%[[m0]], %[[m1]])[%[[mSTRIDE]], %[[mOFF0]], %[[mOFF1]]]
//   CHECK-DAG:           %[[mParts:.+]]:2 = affine.delinearize_index %[[mIDX]] into (32, 32)
//   CHECK-DAG:           %[[hIDX:.+]] = affine.apply #[[$MAP2]](%[[mParts]]#0, %[[kParts]]#0)
//   CHECK-DAG:           %[[wIDX:.+]] = affine.apply #[[$MAP2]](%[[mParts]]#1, %[[kParts]]#1)
//       CHECK:           %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[b]], %[[hIDX]], %[[wIDX]], %[[kParts]]#2] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<2x34x34x640xf32> to tensor<1x1x1x4xf32>
//       CHECK:           %[[OUT_SLICE:.+]] = tensor.extract_slice %[[OUT3]][%[[b]], %[[m0]], %[[m1]], %[[k]], 0] [1, 1, 1, 1, 4] [1, 1, 1, 1, 1] : tensor<2x?x?x2x4xf32> to tensor<1x1x1x4xf32>
//       CHECK:           %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<1x1x1x4xf32>) outs(%[[OUT_SLICE]] : tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
//       CHECK:           %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[OUT3]][%[[b]], %[[m0]], %[[m1]], %[[k]], 0] [1, 1, 1, 1, 4] [1, 1, 1, 1, 1] : tensor<1x1x1x4xf32> into tensor<2x?x?x2x4xf32>
//       CHECK:           scf.yield %[[INSERT]] : tensor<2x?x?x2x4xf32>
//       CHECK:         scf.yield %[[kLOOP]] : tensor<2x?x?x2x4xf32>
//       CHECK:       scf.yield %[[mLOOP1]] : tensor<2x?x?x2x4xf32>
//       CHECK:     scf.yield %[[mLOOP0]] : tensor<2x?x?x2x4xf32>
//       CHECK:   return %[[bLOOP]] : tensor<2x?x?x2x4xf32>

// -----

module {
  func.func @im2col_expanded_nchw(%arg0: tensor<2x640x34x34xf32>, %m0: index, %m1: index, %k: index) -> tensor<2x1x1x2x4xf32> {
    %0 = tensor.empty() : tensor<2x1x1x2x4xf32>
    %7 = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
            m_offset = [%m0, %m1] * [32, 1] k_offset = [%k, 0] * [4, 1]
            batch_pos = [0] m_pos = [1, 2] k_pos = [3]
            input_k_perm = [0, 1, 2]
            ins(%arg0 : tensor<2x640x34x34xf32>)
            outs(%0 : tensor<2x1x1x2x4xf32>) -> tensor<2x1x1x2x4xf32>
    return %7 : tensor<2x1x1x2x4xf32>
  }
}
// Verify that the NCHW layout does not vectorize.
// CHECK-LABEL: func.func @im2col_expanded_nchw
//       CHECK:   linalg.copy ins({{.*}} : tensor<1x1x1x1xf32>) outs({{.*}} : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>

// -----

#map = affine_map<(d0) -> (d0 * 4)>
module {
  func.func @im2col_unrolled(%arg0: tensor<2x34x34x640xf32>, %m_off: index, %k: index) -> tensor<2x2x4xf32> {
    %0 = tensor.empty() : tensor<2x2x4xf32>
    %k_off = affine.apply #map(%k)
    %7 = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
            m_offset = [%m_off] * [1] k_offset = [%k_off] * [1]
            batch_pos = [0] m_pos = [1, 2] k_pos = [3]
            input_k_perm = [0, 1, 2]
            ins(%arg0 : tensor<2x34x34x640xf32>)
            outs(%0 : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
    return %7 : tensor<2x2x4xf32>
  }
}
//   CHECK-UNROLL-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 * 4)>
//   CHECK-UNROLL-DAG: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-UNROLL-LABEL: func.func @im2col_unrolled
//  CHECK-UNROLL-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]
//  CHECK-UNROLL-SAME:     %[[mOFF:[a-zA-Z0-9_]+]]
//  CHECK-UNROLL-SAME:     %[[K:[a-zA-Z0-9_]+]]
//   CHECK-UNROLL-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-UNROLL-DAG:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK-UNROLL:   %[[OUT_TILE:.+]] = tensor.empty() : tensor<2x2x4xf32>

//  First iteration
//
//   CHECK-UNROLL-DAG:   %[[kIDX:.+]] = affine.apply #[[$MAP]]()[%[[K]]]
//   CHECK-UNROLL-DAG:   %[[kParts:.+]]:3 = affine.delinearize_index %[[kIDX]] into (3, 3, 640)
//   CHECK-UNROLL-DAG:   %[[mIDX:.+]] = affine.apply #[[$MAP1]](%[[C0]])[%[[mOFF]]]
//   CHECK-UNROLL-DAG:   %[[mParts:.+]]:2 = affine.delinearize_index %[[mIDX]] into (32, 32)
//   CHECK-UNROLL-DAG:   %[[hIDX:.+]] = affine.apply #[[$MAP1]](%[[mParts]]#0)[%[[kParts]]#0]
//   CHECK-UNROLL-DAG:   %[[wIDX:.+]] = affine.apply #[[$MAP1]](%[[mParts]]#1)[%[[kParts]]#1]
//       CHECK-UNROLL:   %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[C0]], %[[hIDX]], %[[wIDX]], %[[kParts]]#2] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<2x34x34x640xf32> to tensor<1x1x4xf32>
//       CHECK-UNROLL:   %[[OUT_SLICE:.+]] = tensor.extract_slice %[[OUT_TILE]][%[[C0]], %[[C0]], 0] [1, 1, 4] [1, 1, 1] : tensor<2x2x4xf32> to tensor<1x1x4xf32>
//       CHECK-UNROLL:   %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<1x1x4xf32>) outs(%[[OUT_SLICE]] : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
//       CHECK-UNROLL:   %[[INSERT0:.+]] = tensor.insert_slice %[[COPY]] into %[[OUT_TILE]][%[[C0]], %[[C0]], 0] [1, 1, 4] [1, 1, 1] : tensor<1x1x4xf32> into tensor<2x2x4xf32>

//  Second iteration
//
//   CHECK-UNROLL-DAG:   %[[kParts:.+]]:3 = affine.delinearize_index %[[kIDX]] into (3, 3, 640)
//   CHECK-UNROLL-DAG:   %[[mIDX:.+]] = affine.apply #[[$MAP1]](%[[C1]])[%[[mOFF]]]
//   CHECK-UNROLL-DAG:   %[[mParts:.+]]:2 = affine.delinearize_index %[[mIDX]] into (32, 32)
//   CHECK-UNROLL-DAG:   %[[hIDX:.+]] = affine.apply #[[$MAP1]](%[[mParts]]#0)[%[[kParts]]#0]
//   CHECK-UNROLL-DAG:   %[[wIDX:.+]] = affine.apply #[[$MAP1]](%[[mParts]]#1)[%[[kParts]]#1]
//       CHECK-UNROLL:   %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[C0]], %[[hIDX]], %[[wIDX]], %[[kParts]]#2] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<2x34x34x640xf32> to tensor<1x1x4xf32>
//       CHECK-UNROLL:   %[[OUT_SLICE:.+]] = tensor.extract_slice %[[INSERT0]][%[[C0]], %[[C1]], 0] [1, 1, 4] [1, 1, 1] : tensor<2x2x4xf32> to tensor<1x1x4xf32>
//       CHECK-UNROLL:   %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<1x1x4xf32>) outs(%[[OUT_SLICE]] : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
//       CHECK-UNROLL:   %[[INSERT1:.+]] = tensor.insert_slice %[[COPY]] into %[[INSERT0]][%[[C0]], %[[C1]], 0] [1, 1, 4] [1, 1, 1] : tensor<1x1x4xf32> into tensor<2x2x4xf32>

//  Third iteration
//
//   CHECK-UNROLL-DAG:   %[[kParts:.+]]:3 = affine.delinearize_index %[[kIDX]] into (3, 3, 640)
//   CHECK-UNROLL-DAG:   %[[mIDX:.+]] = affine.apply #[[$MAP1]](%[[C0]])[%[[mOFF]]]
//   CHECK-UNROLL-DAG:   %[[mParts:.+]]:2 = affine.delinearize_index %[[mIDX]] into (32, 32)
//   CHECK-UNROLL-DAG:   %[[hIDX:.+]] = affine.apply #[[$MAP1]](%[[mParts]]#0)[%[[kParts]]#0]
//   CHECK-UNROLL-DAG:   %[[wIDX:.+]] = affine.apply #[[$MAP1]](%[[mParts]]#1)[%[[kParts]]#1]
//       CHECK-UNROLL:   %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[C1]], %[[hIDX]], %[[wIDX]], %[[kParts]]#2] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<2x34x34x640xf32> to tensor<1x1x4xf32>
//       CHECK-UNROLL:   %[[OUT_SLICE:.+]] = tensor.extract_slice %[[INSERT1]][%[[C1]], %[[C0]], 0] [1, 1, 4] [1, 1, 1] : tensor<2x2x4xf32> to tensor<1x1x4xf32>
//       CHECK-UNROLL:   %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<1x1x4xf32>) outs(%[[OUT_SLICE]] : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
//       CHECK-UNROLL:   %[[INSERT2:.+]] = tensor.insert_slice %[[COPY]] into %[[INSERT1]][%[[C1]], %[[C0]], 0] [1, 1, 4] [1, 1, 1] : tensor<1x1x4xf32> into tensor<2x2x4xf32>

//  Fourth iteration
//
//   CHECK-UNROLL-DAG:   %[[kParts:.+]]:3 = affine.delinearize_index %[[kIDX]] into (3, 3, 640)
//   CHECK-UNROLL-DAG:   %[[mIDX:.+]] = affine.apply #[[$MAP1]](%[[C1]])[%[[mOFF]]]
//   CHECK-UNROLL-DAG:   %[[mParts:.+]]:2 = affine.delinearize_index %[[mIDX]] into (32, 32)
//   CHECK-UNROLL-DAG:   %[[hIDX:.+]] = affine.apply #[[$MAP1]](%[[mParts]]#0)[%[[kParts]]#0]
//   CHECK-UNROLL-DAG:   %[[wIDX:.+]] = affine.apply #[[$MAP1]](%[[mParts]]#1)[%[[kParts]]#1]
//       CHECK-UNROLL:   %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[C1]], %[[hIDX]], %[[wIDX]], %[[kParts]]#2] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<2x34x34x640xf32> to tensor<1x1x4xf32>
//       CHECK-UNROLL:   %[[OUT_SLICE:.+]] = tensor.extract_slice %[[INSERT2]][%[[C1]], %[[C1]], 0] [1, 1, 4] [1, 1, 1] : tensor<2x2x4xf32> to tensor<1x1x4xf32>
//       CHECK-UNROLL:   %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<1x1x4xf32>) outs(%[[OUT_SLICE]] : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
//       CHECK-UNROLL:   %[[INSERT3:.+]] = tensor.insert_slice %[[COPY]] into %[[INSERT2]][%[[C1]], %[[C1]], 0] [1, 1, 4] [1, 1, 1] : tensor<1x1x4xf32> into tensor<2x2x4xf32>

//       CHECK-UNROLL:   return %[[INSERT3]] : tensor<2x2x4xf32>

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
                              m_offset = [0, 0] * [2, 1] k_offset = [0] * [1]
                              batch_pos = [0] m_pos = [2, 3] k_pos = [1]
                              input_k_perm = [0, 1, 2]
                              ins(%padded : tensor<1x8x9x9xf32>)
                              outs(%empty : tensor<1x2x2x12xf32>) -> tensor<1x2x2x12xf32>
  return %im2col : tensor<1x2x2x12xf32>
  }
}

// CHECK-LABEL: func.func @im2col_padding
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]
//       CHECK: %[[T1:.+]] = tensor.extract_slice %[[ARG0]]
//       CHECK: %[[T2:.+]] = tensor.pad %[[T1]]
//  CHECK-NEXT: ^bb0
//  CHECK-NEXT:   tensor.yield
//  CHECK-NEXT: } : tensor<1x1x?x?xf32> to tensor<1x1x1x1xf32>

// -----

module {
  func.func @im2col_nhc_with_perm(%arg0: tensor<1x3x2xf32>) -> tensor<1x2x4xf32> {
    %0 = tensor.empty() : tensor<1x2x4xf32>
    %1 = iree_linalg_ext.im2col strides = [1] dilations = [1] kernel_size = [2]
                            m_offset = [0] * [1] k_offset = [0] * [1]
                            batch_pos = [0] m_pos = [1] k_pos = [2]
                            input_k_perm = [1, 0]
                            ins(%arg0 : tensor<1x3x2xf32>)
                            outs(%0 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
    return %1 : tensor<1x2x4xf32>
  }
}
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
//   CHECK-DAG:       %[[kParts:.+]]:2 = affine.delinearize_index %[[K]] into (2, 2) : index, index
//   CHECK-DAG:       %[[hIdx:.+]] = affine.apply #[[$MAP]](%[[M]], %[[kParts]]#1)
//       CHECK:       %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[hIdx]], %[[kParts]]#0] [1, 1, 1] [1, 1, 1] : tensor<1x3x2xf32> to tensor<1x1x1xf32>
//       CHECK:       %[[OUT_SLICE:.+]] = tensor.extract_slice %[[OUT2]][0, %[[M]], %[[K]]] [1, 1, 1] [1, 1, 1] : tensor<1x2x4xf32> to tensor<1x1x1xf32>
//       CHECK:       %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<1x1x1xf32>) outs(%[[OUT_SLICE]] : tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
//       CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[OUT2]][0, %[[M]], %[[K]]] [1, 1, 1] [1, 1, 1] : tensor<1x1x1xf32> into tensor<1x2x4xf32>
//       CHECK:       scf.yield %[[INSERT]] : tensor<1x2x4xf32>
//       CHECK:     scf.yield %[[KLOOP]] : tensor<1x2x4xf32>
//       CHECK:   return %[[MLOOP]] : tensor<1x2x4xf32>

// -----

module {
  func.func @im2col_nhwc_with_perm(%arg0: tensor<1x16x16x4xf32>) -> tensor<1x14x14x36xf32> {
    %0 = tensor.empty() : tensor<1x14x14x36xf32>
    %1 = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
                            m_offset = [0, 0] * [14, 1] k_offset = [0] * [1]
                            batch_pos = [0] m_pos = [1, 2] k_pos = [3]
                            input_k_perm = [2, 0, 1]
                            ins(%arg0 : tensor<1x16x16x4xf32>)
                            outs(%0 : tensor<1x14x14x36xf32>) -> tensor<1x14x14x36xf32>
    return %1 : tensor<1x14x14x36xf32>
  }
}
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0 * 14 + d1)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: func.func @im2col_nhwc_with_perm
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x16x16x4xf32>
//   CHECK-DAG: %[[C36:.+]] = arith.constant 36 : index
//   CHECK-DAG: %[[C14:.+]] = arith.constant 14 : index
//   CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//       CHECK: %[[OUT_TILE:.+]] = tensor.empty() : tensor<1x14x14x36xf32>
//       CHECK: %[[MLOOP0:.+]] = scf.for %[[M1:.+]] = %[[C0]] to %[[C14]] step %[[C1]] iter_args(%[[OUT1:.+]] = %[[OUT_TILE]]) -> (tensor<1x14x14x36xf32>)
//       CHECK:   %[[MLOOP1:.+]] = scf.for %[[M2:.+]] = %[[C0]] to %[[C14]] step %[[C1]] iter_args(%[[OUT2:.+]] = %[[OUT1]]) -> (tensor<1x14x14x36xf32>)
//       CHECK:     %[[KLOOP:.+]] = scf.for %[[K:.+]] = %[[C0]] to %[[C36]] step %[[C1]] iter_args(%[[OUT3:.+]] = %[[OUT2]]) -> (tensor<1x14x14x36xf32>)
//   CHECK-DAG:       %[[kParts:.+]]:3 = affine.delinearize_index %[[K]] into (4, 3, 3) : index, index, index
//   CHECK-DAG:       %[[FLAT_M:.+]] = affine.apply #[[$MAP]](%[[M1]], %[[M2]])
//   CHECK-DAG:       %[[mParts:.+]]:2 = affine.delinearize_index %[[FLAT_M]] into (14, 14) : index, index
//   CHECK-DAG:       %[[hIDX:.+]] = affine.apply #[[$MAP1]](%[[mParts]]#0, %[[kParts]]#1)
//   CHECK-DAG:       %[[wIDX:.+]] = affine.apply #[[$MAP1]](%[[mParts]]#1, %[[kParts]]#2)
//       CHECK:       %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[hIDX]], %[[wIDX]], %[[kParts]]#0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x16x16x4xf32> to tensor<1x1x1x1xf32>
//       CHECK:       %[[OUT_SLICE:.+]] = tensor.extract_slice %[[OUT3]][0, %[[M1]], %[[M2]], %[[K]]]
//       CHECK:       %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<1x1x1x1xf32>) outs(%[[OUT_SLICE]] : tensor<1x1x1x1xf32>)
//       CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[OUT3]][0, %[[M1]], %[[M2]], %[[K]]]
//       CHECK:     scf.yield %[[INSERT]] : tensor<1x14x14x36xf32>
//       CHECK:   scf.yield %[[KLOOP]] : tensor<1x14x14x36xf32>
//       CHECK: scf.yield %[[MLOOP1]] : tensor<1x14x14x36xf32>
//       CHECK: return %[[MLOOP0]] : tensor<1x14x14x36xf32>

// -----

module {
  func.func @im2col_chwn(%arg0: tensor<16x26x18x4xf32>, %arg1: index, %arg2: index, %arg3: index) -> tensor<4x2x2x2xf32> {
    %0 = tensor.empty() : tensor<4x2x2x2xf32>
    %1 = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [1, 1] kernel_size = [24, 16]
            m_offset = [%arg1, %arg2] * [3, 1] k_offset = [%arg3] * [1]
            batch_pos = [3] m_pos = [1, 2] k_pos = [0]
            input_k_perm = [0, 1, 2]
            ins(%arg0 : tensor<16x26x18x4xf32>)
            outs(%0 : tensor<4x2x2x2xf32>) -> tensor<4x2x2x2xf32>
    return %1 : tensor<4x2x2x2xf32>
  }
}

//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * 3 + d1 + s0 * 3 + s1)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
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
//   CHECK-DAG:       %[[mIDX:.+]] = affine.apply #[[$MAP1]](%[[M0]], %[[M1]])[%[[ARG1]], %[[ARG2]]]
//   CHECK-DAG:       %[[mParts:.+]]:2 = affine.delinearize_index %[[mIDX]] into (3, 3)
//   CHECK-DAG:       %[[hIDX:.+]] = affine.apply #[[$MAP2]](%[[mParts]]#0, %[[kParts]]#1)
//   CHECK-DAG:       %[[wIDX:.+]] = affine.apply #[[$MAP2]](%[[mParts]]#1, %[[kParts]]#2)
//       CHECK:       %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[kParts]]#0, %[[hIDX]], %[[wIDX]], 0] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<16x26x18x4xf32> to tensor<1x1x1x4xf32>
//       CHECK:       %[[EMPTY:.+]] = tensor.empty() : tensor<4x1x1x1xf32>
//       CHECK:       %[[TRANS:.+]] = linalg.transpose ins(%[[IN_SLICE]] : tensor<1x1x1x4xf32>) outs(%[[EMPTY]] : tensor<4x1x1x1xf32>) permutation = [3, 1, 2, 0]
//       CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[TRANS]] into %[[OUT2]][0, %[[M0]], %[[M1]], %[[K]]] [4, 1, 1, 1] [1, 1, 1, 1] : tensor<4x1x1x1xf32> into tensor<4x2x2x2xf32>
//       CHECK:      scf.yield %[[INSERT]] : tensor<4x2x2x2xf32>
//       CHECK:    scf.yield %[[kLOOP]] : tensor<4x2x2x2xf32>
//       CHECK:  scf.yield %[[mLOOP1]] : tensor<4x2x2x2xf32>
//       CHECK: return %[[mLOOP0:.+]] : tensor<4x2x2x2xf32>

// -----

module {
  func.func @im2col_chwn_rank_reduce(%arg0: tensor<16x26x18x4xf32>, %arg1: index, %arg2: index, %m_size: index, %k_size: index) -> tensor<4x?x?xf32> {
    %0 = tensor.empty(%m_size, %k_size) : tensor<4x?x?xf32>
    %1 = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [1, 1] kernel_size = [24, 16]
            m_offset = [%arg1] * [1] k_offset = [%arg2] * [1]
            batch_pos = [3] m_pos = [1, 2] k_pos = [0]
            input_k_perm = [0, 1, 2]
            ins(%arg0 : tensor<16x26x18x4xf32>)
            outs(%0 : tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    return %1 : tensor<4x?x?xf32>
  }
}

// Verify that when the batch dimension is the innermost and generates rank-reduced output,
// a 1d tensor slice is extracted and transpose is not needed.
// CHECK-LABEL: func.func @im2col_chwn_rank_reduce
//       CHECK:     %[[IN_SLICE:.+]] = tensor.extract_slice {{.*}} : tensor<16x26x18x4xf32> to tensor<4xf32>
//       CHECK:     %[[OUT_SLICE:.+]] = tensor.extract_slice {{.*}} : tensor<4x?x?xf32> to tensor<4xf32>
//       CHECK:     %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<4xf32>) outs(%[[OUT_SLICE]] : tensor<4xf32>)
//       CHECK:     %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into {{.*}} : tensor<4xf32> into tensor<4x?x?xf32>
