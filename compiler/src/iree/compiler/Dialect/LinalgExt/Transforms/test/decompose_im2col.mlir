// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-im2col))" --split-input-file %s | FileCheck %s

#map = affine_map<(d0) -> (d0 * 5)>
#map1 = affine_map<(d0) -> (d0 * 4)>
#map2 = affine_map<(d0) -> (-d0 + 1024, 5)>
module {
  func.func @im2col(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1024x5760xf32> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<2x1024x5760xf32>
    %1 = scf.for %arg1 = %c0 to %c2 step %c1 iter_args(%arg2 = %0) -> (tensor<2x1024x5760xf32>) {
      %c205 = arith.constant 205 : index
      %2 = scf.for %arg3 = %c0 to %c205 step %c1 iter_args(%arg4 = %arg2) -> (tensor<2x1024x5760xf32>) {
        %3 = affine.apply #map(%arg3)
        %c1440 = arith.constant 1440 : index
        %4 = scf.for %arg5 = %c0 to %c1440 step %c1 iter_args(%arg6 = %arg4) -> (tensor<2x1024x5760xf32>) {
          %5 = affine.apply #map1(%arg5)
          %6 = affine.min #map2(%3)
          %extracted_slice = tensor.extract_slice %arg0[%arg1, 0, 0, 0] [1, 34, 34, 640] [1, 1, 1, 1] : tensor<2x34x34x640xf32> to tensor<1x34x34x640xf32>
          %extracted_slice_0 = tensor.extract_slice %arg6[%arg1, %3, %5] [1, %6, 4] [1, 1, 1] : tensor<2x1024x5760xf32> to tensor<1x?x4xf32>
          %7 = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3] m_offset = [%3] k_offset = [%5] batch_pos = [0] m_pos = [1, 2] k_pos = [3] ins(%extracted_slice : tensor<1x34x34x640xf32>) outs(%extracted_slice_0 : tensor<1x?x4xf32>) -> tensor<1x?x4xf32>
          %inserted_slice = tensor.insert_slice %7 into %arg6[%arg1, %3, %5] [1, %6, 4] [1, 1, 1] : tensor<1x?x4xf32> into tensor<2x1024x5760xf32>
          scf.yield %inserted_slice : tensor<2x1024x5760xf32>
        }
        scf.yield %4 : tensor<2x1024x5760xf32>
      }
      scf.yield %2 : tensor<2x1024x5760xf32>
    }
    return %1 : tensor<2x1024x5760xf32>
  }
}
// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0) -> (d0 * 5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0) -> (-d0 + 1024, 5)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 160) * 640)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> ((d0 + d1 * 5) floordiv 32 + d2 floordiv 480)>
// CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> ((d0 + d1 * 5) mod 32 + d2 floordiv 160 - (d2 floordiv 480) * 3)>
//     CHECK: func.func @im2col(%[[ARG0:.+]]: tensor<2x34x34x640xf32>)
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//     CHECK:   scf.for %[[B:.+]] = {{.*}} to {{.*}}
//     CHECK:     scf.for %[[M:.+]] = {{.*}} to {{.*}}
//     CHECK:       scf.for %[[K:.+]] = {{.*}} to {{.*}}
//     CHECK:         %[[mSIZE:.+]] = affine.min #[[MAP2]]
//     CHECK:         %[[IN_TILE:.+]] = tensor.extract_slice %[[ARG0]]
//     CHECK:         %[[OUT_TILE:.+]] = tensor.extract_slice
//     CHECK:         %[[bLOOP:.+]] = scf.for %[[b:.+]] = %[[C0]] to %[[C1]] step %[[C1]] iter_args(%[[OUT0:.+]] = %[[OUT_TILE]]) -> (tensor<1x?x4xf32>) {
//     CHECK:           %[[mLOOP:.+]] = scf.for %[[m:.+]] = %[[C0]] to %[[mSIZE]] step %[[C1]] iter_args(%[[OUT1:.+]] = %[[OUT0]]) -> (tensor<1x?x4xf32>) {
// CHECK-DAG:             %[[kIDX:.+]] = affine.apply #[[MAP3]](%[[K]])
// CHECK-DAG:             %[[hIDX:.+]] = affine.apply #[[MAP4]](%[[m]], %[[M]], %[[K]])
// CHECK-DAG:             %[[wIDX:.+]] = affine.apply #[[MAP5]](%[[m]], %[[M]], %[[K]])
//     CHECK:             %[[IN_SLICE:.+]] = tensor.extract_slice %[[IN_TILE]][%[[b]], %[[hIDX]], %[[wIDX]], %[[kIDX]]] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<1x34x34x640xf32> to tensor<4xf32>
//     CHECK:             %[[OUT_SLICE:.+]] = tensor.extract_slice %[[OUT1]][%[[b]], %[[m]], 0] [1, 1, 4] [1, 1, 1] : tensor<1x?x4xf32> to tensor<4xf32>
//     CHECK:             %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<4xf32>) outs(%[[OUT_SLICE]] : tensor<4xf32>) -> tensor<4xf32>
//     CHECK:             %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[OUT1]][%[[b]], %[[m]], 0] [1, 1, 4] [1, 1, 1] : tensor<4xf32> into tensor<1x?x4xf32>
//     CHECK:             scf.yield %[[INSERT]] : tensor<1x?x4xf32>
//     CHECK:           }
//     CHECK:           scf.yield %[[mLOOP]] : tensor<1x?x4xf32>
//     CHECK:         }
//     CHECK:         %[[INSERT_TILE:.+]] = tensor.insert_slice %[[bLOOP]] {{.*}} : tensor<1x?x4xf32> into tensor<2x1024x5760xf32>
//     CHECK:         scf.yield %[[INSERT_TILE]] : tensor<2x1024x5760xf32>
//     CHECK:       }
//     CHECK:       scf.yield
//     CHECK:     }
//     CHECK:     scf.yield
//     CHECK:   }

// -----

#map = affine_map<(d0) -> (d0 * 9)>
#map1 = affine_map<(d0) -> (d0 * 7)>
#map2 = affine_map<(d0) -> (-d0 + 1024, 9)>
#map3 = affine_map<(d0) -> (-d0 + 5760, 7)>
module {
  func.func @im2col_transposed_m_pos(%arg0: tensor<640x2x101x172xf32>) -> tensor<2x1024x5760xf32> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<2x1024x5760xf32>
    %1 = scf.for %arg1 = %c0 to %c2 step %c1 iter_args(%arg2 = %0) -> (tensor<2x1024x5760xf32>) {
      %c114 = arith.constant 114 : index
      %2 = scf.for %arg3 = %c0 to %c114 step %c1 iter_args(%arg4 = %arg2) -> (tensor<2x1024x5760xf32>) {
        %3 = affine.apply #map(%arg3)
        %c823 = arith.constant 823 : index
        %4 = scf.for %arg5 = %c0 to %c823 step %c1 iter_args(%arg6 = %arg4) -> (tensor<2x1024x5760xf32>) {
          %5 = affine.apply #map1(%arg5)
          %6 = affine.min #map2(%3)
          %7 = affine.min #map3(%5)
          %extracted_slice = tensor.extract_slice %arg0[0, %arg1, 0, 0] [640, 1, 101, 172] [1, 1, 1, 1] : tensor<640x2x101x172xf32> to tensor<640x1x101x172xf32>
          %extracted_slice_0 = tensor.extract_slice %arg6[%arg1, %3, %5] [1, %6, %7] [1, 1, 1] : tensor<2x1024x5760xf32> to tensor<1x?x?xf32>
          %8 = iree_linalg_ext.im2col strides = [5, 3] dilations = [4, 7] kernel_size = [5, 2] m_offset = [%3] k_offset = [%5] batch_pos = [1] m_pos = [3, 2] k_pos = [0] ins(%extracted_slice : tensor<640x1x101x172xf32>) outs(%extracted_slice_0 : tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
          %inserted_slice = tensor.insert_slice %8 into %arg6[%arg1, %3, %5] [1, %6, %7] [1, 1, 1] : tensor<1x?x?xf32> into tensor<2x1024x5760xf32>
          scf.yield %inserted_slice : tensor<2x1024x5760xf32>
        }
        scf.yield %4 : tensor<2x1024x5760xf32>
      }
      scf.yield %2 : tensor<2x1024x5760xf32>
    }
    return %1 : tensor<2x1024x5760xf32>
  }
}
// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0) -> (d0 * 9)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> (d0 * 7)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0) -> (-d0 + 1024, 9)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0) -> (-d0 + 5760, 7)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1) -> ((d0 + d1 * 7) floordiv 10)>
// CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2, d3) -> (((d0 + d1 * 9) floordiv 32) * 5 + (((d2 + d3 * 7) mod 10) floordiv 5) * 4)>
// CHECK-DAG: #[[MAP6:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 * 3 + d1 * 27 + d2 * 7 + d3 * 49 - ((d0 + d1 * 9) floordiv 32) * 96 - ((d2 + d3 * 7) floordiv 5) * 35)>
//     CHECK: func.func @im2col_transposed_m_pos(%[[ARG0:.+]]: tensor<640x2x101x172xf32>)
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//     CHECK:   scf.for %[[B:.+]] = {{.*}} to {{.*}}
//     CHECK:     scf.for %[[M:.+]] = {{.*}} to {{.*}}
//     CHECK:       scf.for %[[K:.+]] = {{.*}} to {{.*}}
//     CHECK:         %[[mSIZE:.+]] = affine.min #[[MAP2]]
//     CHECK:         %[[kSIZE:.+]] = affine.min #[[MAP3]]
//     CHECK:         %[[IN_TILE:.+]] = tensor.extract_slice %[[ARG0]]
//     CHECK:         %[[OUT_TILE:.+]] = tensor.extract_slice
//     CHECK:         %[[bLOOP:.+]] = scf.for %[[b:.+]] = %[[C0]] to %[[C1]] step %[[C1]] iter_args(%[[OUT0:.+]] = %[[OUT_TILE]]) -> (tensor<1x?x?xf32>) {
//     CHECK:           %[[mLOOP:.+]] = scf.for %[[m:.+]] = %[[C0]] to %[[mSIZE]] step %[[C1]] iter_args(%[[OUT1:.+]] = %[[OUT0]]) -> (tensor<1x?x?xf32>) {
//     CHECK:             %[[kLOOP:.+]] = scf.for %[[k:.+]] = %[[C0]] to %[[kSIZE]] step %[[C1]] iter_args(%[[OUT2:.+]] = %[[OUT1]]) -> (tensor<1x?x?xf32>) {
// CHECK-DAG:               %[[kIDX:.+]] = affine.apply #[[MAP4]](%[[k]], %[[K]])
// CHECK-DAG:               %[[hIDX:.+]] = affine.apply #[[MAP5]](%[[m]], %[[M]], %[[k]], %[[K]])
// CHECK-DAG:               %[[wIDX:.+]] = affine.apply #[[MAP6]](%[[m]], %[[M]], %[[k]], %[[K]])
//     CHECK:               %[[IN_SLICE:.+]] = tensor.extract_slice %[[IN_TILE]][%[[kIDX]], %[[b]], %[[wIDX]], %[[hIDX]]] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<640x1x101x172xf32> to tensor<1xf32>
//     CHECK:               %[[OUT_SLICE:.+]] = tensor.extract_slice %[[OUT2]][%[[b]], %[[m]], %[[k]]] [1, 1, 1] [1, 1, 1] : tensor<1x?x?xf32> to tensor<1xf32>
//     CHECK:               %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<1xf32>) outs(%[[OUT_SLICE]] : tensor<1xf32>) -> tensor<1xf32>
//     CHECK:               %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[OUT2]][%[[b]], %[[m]], %[[k]]] [1, 1, 1] [1, 1, 1] : tensor<1xf32> into tensor<1x?x?xf32>
//     CHECK:               scf.yield %[[INSERT]] : tensor<1x?x?xf32>
//     CHECK:             }
//     CHECK:             scf.yield %[[kLOOP]] : tensor<1x?x?xf32>
//     CHECK:           }
//     CHECK:           scf.yield %[[mLOOP]] : tensor<1x?x?xf32>
//     CHECK:         }
//     CHECK:         %[[INSERT_TILE:.+]] = tensor.insert_slice %[[bLOOP]] {{.*}} : tensor<1x?x?xf32> into tensor<2x1024x5760xf32>
//     CHECK:         scf.yield %[[INSERT_TILE]] : tensor<2x1024x5760xf32>
//     CHECK:       }
//     CHECK:       scf.yield
//     CHECK:     }
//     CHECK:     scf.yield
//     CHECK:   }
