// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-im2col))" --split-input-file %s | FileCheck %s

#map = affine_map<(d0) -> (d0 * 4)>
module {
  func.func @im2col_untile_k(%arg0: tensor<2x34x34x640xf32>, %m_size: index, %m_off: index, %k: index) -> tensor<2x?x4xf32> {
    %0 = tensor.empty(%m_size) : tensor<2x?x4xf32>
    %k_off = affine.apply #map(%k)
    %7 = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3] m_offset = [%m_off] k_offset = [%k_off] batch_pos = [0] m_pos = [1, 2] k_pos = [3] ins(%arg0 : tensor<2x34x34x640xf32>) outs(%0 : tensor<2x?x4xf32>) -> tensor<2x?x4xf32>
    return %7 : tensor<2x?x4xf32>
  }
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 160) * 640)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> ((d0 + s0) floordiv 32 + s1 floordiv 480)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0, s1] -> ((d0 + s0) mod 32 + s1 floordiv 160 - (s1 floordiv 480) * 3)>
//      CHECK: func.func @im2col_untile_k(%[[ARG0:.+]]: tensor<2x34x34x640xf32>
// CHECK-SAME:   %[[mSIZE:.+]]: index, %[[mOFF:.+]]: index, %[[K:.+]]: index)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//      CHECK:   %[[OUT_TILE:.+]] = tensor.empty(%[[mSIZE]]) : tensor<2x?x4xf32>
//      CHECK:   %[[bLOOP:.+]] = scf.for %[[b:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUT0:.+]] = %[[OUT_TILE]]) -> (tensor<2x?x4xf32>) {
//      CHECK:     %[[mLOOP:.+]] = scf.for %[[m:.+]] = %[[C0]] to %[[mSIZE]] step %[[C1]] iter_args(%[[OUT1:.+]] = %[[OUT0]]) -> (tensor<2x?x4xf32>) {
//  CHECK-DAG:       %[[kIDX:.+]] = affine.apply #[[MAP]]()[%[[K]]]
//  CHECK-DAG:       %[[hIDX:.+]] = affine.apply #[[MAP1]](%[[m]])[%[[mOFF]], %[[K]]]
//  CHECK-DAG:       %[[wIDX:.+]] = affine.apply #[[MAP2]](%[[m]])[%[[mOFF]], %[[K]]]
//      CHECK:       %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[b]], %[[hIDX]], %[[wIDX]], %[[kIDX]]] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<2x34x34x640xf32> to tensor<4xf32>
//      CHECK:       %[[OUT_SLICE:.+]] = tensor.extract_slice %[[OUT1]][%[[b]], %[[m]], 0] [1, 1, 4] [1, 1, 1] : tensor<2x?x4xf32> to tensor<4xf32>
//      CHECK:       %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<4xf32>) outs(%[[OUT_SLICE]] : tensor<4xf32>) -> tensor<4xf32>
//      CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[OUT1]][%[[b]], %[[m]], 0] [1, 1, 4] [1, 1, 1] : tensor<4xf32> into tensor<2x?x4xf32>
//      CHECK:       scf.yield %[[INSERT]] : tensor<2x?x4xf32>
//      CHECK:     }
//      CHECK:     scf.yield %[[mLOOP]] : tensor<2x?x4xf32>
//      CHECK:   }
//      CHECK:   return %[[bLOOP]] : tensor<2x?x4xf32>

// -----

module {
  func.func @im2col_transposed_m_pos(%arg0: tensor<640x2x101x172xf32>, %m_size: index, %k_size: index, %m_off: index, %k_off: index) -> tensor<2x?x?xf32> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty(%m_size, %k_size) : tensor<2x?x?xf32>
    %8 = iree_linalg_ext.im2col strides = [5, 3] dilations = [4, 7] kernel_size = [5, 2] m_offset = [%m_off] k_offset = [%k_off] batch_pos = [1] m_pos = [3, 2] k_pos = [0] ins(%arg0 : tensor<640x2x101x172xf32>) outs(%0 : tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
    return %8 : tensor<2x?x?xf32>
  }
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0)[s0] -> ((d0 + s0) floordiv 10)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1)[s0, s1] -> (((d0 + s0) floordiv 32) * 5 + (((d1 + s1) mod 10) floordiv 5) * 4)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * 3 + d1 * 7 + s0 * 3 + s1 * 7 - ((d0 + s0) floordiv 32) * 96 - ((d1 + s1) floordiv 5) * 35)>
//      CHECK: func.func @im2col_transposed_m_pos(%[[ARG0:.+]]: tensor<640x2x101x172xf32>
// CHECK-SAME:   %[[mSIZE:.+]]: index, %[[kSIZE:.+]]: index, %[[mOFF:.+]]: index, %[[kOFF:.+]]: index)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//      CHECK:   %[[OUT_TILE:.+]] = tensor.empty(%[[mSIZE]], %[[kSIZE]]) : tensor<2x?x?xf32>
//      CHECK:   %[[bLOOP:.+]] = scf.for %[[b:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[OUT0:.+]] = %[[OUT_TILE]]) -> (tensor<2x?x?xf32>) {
//      CHECK:     %[[mLOOP:.+]] = scf.for %[[m:.+]] = %[[C0]] to %[[mSIZE]] step %[[C1]] iter_args(%[[OUT1:.+]] = %[[OUT0]]) -> (tensor<2x?x?xf32>) {
//      CHECK:       %[[kLOOP:.+]] = scf.for %[[k:.+]] = %[[C0]] to %[[kSIZE]] step %[[C1]] iter_args(%[[OUT2:.+]] = %[[OUT1]]) -> (tensor<2x?x?xf32>) {
//  CHECK-DAG:         %[[kIDX:.+]] = affine.apply #[[MAP]](%[[k]])[%[[kOFF]]]
//  CHECK-DAG:         %[[hIDX:.+]] = affine.apply #[[MAP1]](%[[m]], %[[k]])[%[[mOFF]], %[[kOFF]]]
//  CHECK-DAG:         %[[wIDX:.+]] = affine.apply #[[MAP2]](%[[m]], %[[k]])[%[[mOFF]], %[[kOFF]]]
//      CHECK:         %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[kIDX]], %[[b]], %[[wIDX]], %[[hIDX]]] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<640x2x101x172xf32> to tensor<1xf32>
//      CHECK:         %[[OUT_SLICE:.+]] = tensor.extract_slice %[[OUT2]][%[[b]], %[[m]], %[[k]]] [1, 1, 1] [1, 1, 1] : tensor<2x?x?xf32> to tensor<1xf32>
//      CHECK:         %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<1xf32>) outs(%[[OUT_SLICE]] : tensor<1xf32>) -> tensor<1xf32>
//      CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[OUT2]][%[[b]], %[[m]], %[[k]]] [1, 1, 1] [1, 1, 1] : tensor<1xf32> into tensor<2x?x?xf32>
//      CHECK:         scf.yield %[[INSERT]] : tensor<2x?x?xf32>
//      CHECK:       }
//      CHECK:       scf.yield %[[kLOOP]] : tensor<2x?x?xf32>
//      CHECK:     }
//      CHECK:     scf.yield %[[mLOOP]] : tensor<2x?x?xf32>
//      CHECK:   }
//      CHECK:   return %[[bLOOP]] : tensor<2x?x?xf32>
