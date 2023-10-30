// RUN: iree-opt --iree-global-opt-lift-generic-to-tranpose-batch-matmul --split-input-file %s | FileCheck %s

module {
  func.func @raise_batch_vecmat(%arg0: tensor<32x128xi16>, %arg1: tensor<11008x32x128xi4>) -> tensor<11008x32xi32> {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<11008x32xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<11008x32xi32>) -> tensor<11008x32xi32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, 
                                          affine_map<(d0, d1, d2) -> (d0, d1, d2)>, 
                                          affine_map<(d0, d1, d2) -> (d0, d1)>], 
                         iterator_types = ["parallel", "parallel", "reduction"]} 
                         ins(%arg0, %arg1 : tensor<32x128xi16>, tensor<11008x32x128xi4>) 
                         outs(%1 : tensor<11008x32xi32>) {
    ^bb0(%in: i16, %in_0: i4, %out: i32):
      %3 = arith.extsi %in : i16 to i32
      %4 = arith.extui %in_0 : i4 to i32
      %5 = arith.muli %3, %4 : i32
      %6 = arith.addi %5, %out : i32
      linalg.yield %6 : i32
    } -> tensor<11008x32xi32>
    return %2 : tensor<11008x32xi32>
  }
}

//      CHECK:  func @raise_batch_vecmat(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<32x128xi16>, %[[ARG1:.+]]: tensor<11008x32x128xi4>
//  CHECK-DAG:  %[[CST:.+]] = arith.constant 0 : i32
//  CHECK-DAG:  %[[INIT0:.+]] = tensor.empty() : tensor<32x11008xi32>
//  CHECK-DAG:  %[[FILL:.+]] = linalg.fill ins(%[[CST]] : i32) outs(%[[INIT0]] : tensor<32x11008xi32>)
//  CHECK-DAG:  %[[INIT1:.+]] = tensor.empty() : tensor<32x128x11008xi4>
//  CHECK-DAG:  %[[TRANSPOSE0:.+]] = linalg.transpose ins(%[[ARG1]] : tensor<11008x32x128xi4>) outs(%[[INIT1]] : tensor<32x128x11008xi4>) permutation = [1, 2, 0]
//  CHECK-DAG:  %[[EXTSI:.+]] = arith.extsi %[[ARG0]] : tensor<32x128xi16> to tensor<32x128xi32>
//  CHECK-DAG:  %[[EXTUI:.+]] = arith.extui %[[TRANSPOSE0]] : tensor<32x128x11008xi4> to tensor<32x128x11008xi32>
//      CHECK:  %[[VECMAT:.+]] = linalg.batch_vecmat ins(%[[EXTSI]], %[[EXTUI]] : tensor<32x128xi32>, tensor<32x128x11008xi32>) outs(%[[FILL]] : tensor<32x11008xi32>)
//      CHECK:  %[[INIT2:.+]] = tensor.empty() : tensor<11008x32xi32>
//      CHECK:  %[[TRANSPOSE1:.+]] = linalg.transpose ins(%[[VECMAT]] : tensor<32x11008xi32>) outs(%[[INIT2]] : tensor<11008x32xi32>) permutation = [1, 0]
//      CHECK:  return %[[TRANSPOSE1]]