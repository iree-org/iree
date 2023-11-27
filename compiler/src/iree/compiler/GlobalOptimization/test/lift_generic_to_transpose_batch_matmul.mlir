// RUN: iree-opt --iree-global-opt-lift-generic-to-tranpose-batch-matmul --canonicalize --cse --split-input-file %s | FileCheck %s

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
//  CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//      CHECK:  func @raise_batch_vecmat(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<32x128xi16>, %[[ARG1:.+]]: tensor<11008x32x128xi4>
//  CHECK-DAG:  %[[CST:.+]] = arith.constant 0 : i32
//  CHECK-DAG:  %[[INIT1:.+]] = tensor.empty() : tensor<32x128x11008xi4>
//  CHECK-DAG:  %[[TRANSPOSE0:.+]] = linalg.transpose ins(%[[ARG1]] : tensor<11008x32x128xi4>) outs(%[[INIT1]] : tensor<32x128x11008xi4>) permutation = [1, 2, 0]
//      CHECK:  %[[INIT_EXTSI:.+]] = tensor.empty() : tensor<32x128xi32>
//      CHECK:  %[[EXTSI:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP0]]], iterator_types = ["parallel", "parallel"]} ins(%[[ARG0]] : tensor<32x128xi16>) outs(%[[INIT_EXTSI]] : tensor<32x128xi32>) {
// CHECK-NEXT:     ^bb0(%[[EXTSI_ARG_IN:.+]]: i16, %[[EXTSI_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[EXTSI_OP:.+]] = arith.extsi %[[EXTSI_ARG_IN]] : i16 to i32
// CHECK-NEXT:     linalg.yield %[[EXTSI_OP]] : i32
//      CHECK:  %[[INIT_EXTUI:.+]] = tensor.empty() : tensor<32x128x11008xi32>
//      CHECK:  %[[EXTUI:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[TRANSPOSE0]] : tensor<32x128x11008xi4>) outs(%[[INIT_EXTUI]] : tensor<32x128x11008xi32>) {
// CHECK-NEXT:     ^bb0(%[[EXTUI_ARG_IN:.+]]: i4, %[[EXTUI_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[EXTUI_OP:.+]] = arith.extui %[[EXTUI_ARG_IN]] : i4 to i32
// CHECK-NEXT:     linalg.yield %[[EXTUI_OP]] : i32
//      CHECK:  %[[INIT0:.+]] = tensor.empty() : tensor<32x11008xi32>
//      CHECK:  %[[FILL:.+]] = linalg.fill ins(%[[CST]] : i32) outs(%[[INIT0]] : tensor<32x11008xi32>)
//      CHECK:  %[[VECMAT:.+]] = linalg.batch_vecmat ins(%[[EXTSI]], %[[EXTUI]] : tensor<32x128xi32>, tensor<32x128x11008xi32>) outs(%[[FILL]] : tensor<32x11008xi32>)
//      CHECK:  %[[INIT2:.+]] = tensor.empty() : tensor<11008x32xi32>
//      CHECK:  %[[TRANSPOSE1:.+]] = linalg.transpose ins(%[[VECMAT]] : tensor<32x11008xi32>) outs(%[[INIT2]] : tensor<11008x32xi32>) permutation = [1, 0]
//      CHECK:  return %[[TRANSPOSE1]]

// -----

module {
  func.func @raise_batch_matvec(%arg0: tensor<11008x32x128xi4>, %arg1: tensor<128x32xi16>) -> tensor<11008x32xi32> {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<11008x32xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<11008x32xi32>) -> tensor<11008x32xi32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, 
                                          affine_map<(d0, d1, d2) -> (d2, d1)>, 
                                          affine_map<(d0, d1, d2) -> (d0, d1)>], 
                         iterator_types = ["parallel", "parallel", "reduction"]} 
                         ins(%arg0, %arg1 : tensor<11008x32x128xi4>, tensor<128x32xi16>) 
                         outs(%1 : tensor<11008x32xi32>) {
    ^bb0(%in: i4, %in_0: i16, %out: i32):
      %3 = arith.extui %in : i4 to i32
      %4 = arith.extsi %in_0 : i16 to i32
      %5 = arith.muli %3, %4 : i32
      %6 = arith.addi %5, %out : i32
      linalg.yield %6 : i32
    } -> tensor<11008x32xi32>
    return %2 : tensor<11008x32xi32>
  }
}
//  CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//      CHECK:  func @raise_batch_matvec(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<11008x32x128xi4>, %[[ARG1:.+]]: tensor<128x32xi16>
//  CHECK-DAG:  %[[CST:.+]] = arith.constant 0 : i32
//  CHECK-DAG:  %[[INIT1:.+]] = tensor.empty() : tensor<32x11008x128xi4>
//  CHECK-DAG:  %[[TRANSPOSE0:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<11008x32x128xi4>) outs(%[[INIT1]] : tensor<32x11008x128xi4>) permutation = [1, 0, 2]
//  CHECK-DAG:  %[[INIT2:.+]] = tensor.empty() : tensor<32x128xi16>
//  CHECK-DAG:  %[[TRANSPOSE1:.+]] = linalg.transpose ins(%[[ARG1]] : tensor<128x32xi16>) outs(%[[INIT2]] : tensor<32x128xi16>) permutation = [1, 0]
//      CHECK:  %[[INIT_EXTUI:.+]] = tensor.empty() : tensor<32x11008x128xi32>
//      CHECK:  %[[EXTUI:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP0]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[TRANSPOSE0]] : tensor<32x11008x128xi4>) outs(%[[INIT_EXTUI]] : tensor<32x11008x128xi32>) {
// CHECK-NEXT:     ^bb0(%[[EXTUI_ARG_IN:.+]]: i4, %[[EXTUI_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[EXTUI_OP:.+]] = arith.extui %[[EXTUI_ARG_IN]] : i4 to i32
// CHECK-NEXT:     linalg.yield %[[EXTUI_OP]] : i32
//      CHECK:  %[[INIT_EXTSI:.+]] = tensor.empty() : tensor<32x128xi32>
//      CHECK:  %[[EXTSI:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[TRANSPOSE1]] : tensor<32x128xi16>) outs(%[[INIT_EXTSI]] : tensor<32x128xi32>) {
// CHECK-NEXT:     ^bb0(%[[EXTSI_ARG_IN:.+]]: i16, %[[EXTSI_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[EXTSI_OP:.+]] = arith.extsi %[[EXTSI_ARG_IN]] : i16 to i32
// CHECK-NEXT:     linalg.yield %[[EXTSI_OP]] : i32
//      CHECK:  %[[INIT0:.+]] = tensor.empty() : tensor<32x11008xi32>
//      CHECK:  %[[FILL:.+]] = linalg.fill ins(%[[CST]] : i32) outs(%[[INIT0]] : tensor<32x11008xi32>)
//      CHECK:  %[[MATMUL:.+]] = linalg.batch_matvec ins(%[[EXTUI]], %[[EXTSI]] : tensor<32x11008x128xi32>, tensor<32x128xi32>) outs(%[[FILL]] : tensor<32x11008xi32>)
//      CHECK:  %[[INIT3:.+]] = tensor.empty() : tensor<11008x32xi32>
//      CHECK:  %[[TRANSPOSE2:.+]] = linalg.transpose ins(%[[MATMUL]] : tensor<32x11008xi32>) outs(%[[INIT3]] : tensor<11008x32xi32>) permutation = [1, 0]
//      CHECK:  return %[[TRANSPOSE2]]

// -----

module {
  func.func @raise_batch_matmul(%arg0: tensor<8x32x128xi16>, %arg1: tensor<11008x32x128xi4>) -> tensor<11008x32x8xi32> {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<11008x32x8xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<11008x32x8xi32>) -> tensor<11008x32x8xi32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>, 
                                          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, 
                                          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>], 
                         iterator_types = ["parallel", "parallel", "reduction", "parallel"]} 
                         ins(%arg0, %arg1 : tensor<8x32x128xi16>, tensor<11008x32x128xi4>) 
                         outs(%1 : tensor<11008x32x8xi32>) {
    ^bb0(%in: i16, %in_0: i4, %out: i32):
      %3 = arith.extsi %in : i16 to i32
      %4 = arith.extui %in_0 : i4 to i32
      %5 = arith.muli %3, %4 : i32
      %6 = arith.addi %5, %out : i32
      linalg.yield %6 : i32
    } -> tensor<11008x32x8xi32>
    return %2 : tensor<11008x32x8xi32>
  }
}
//  CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//      CHECK:  func @raise_batch_matmul(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<8x32x128xi16>, %[[ARG1:.+]]: tensor<11008x32x128xi4>
//  CHECK-DAG:  %[[CST:.+]] = arith.constant 0 : i32
//  CHECK-DAG:  %[[INIT1:.+]] = tensor.empty() : tensor<32x8x128xi16>
//  CHECK-DAG:  %[[TRANSPOSE0:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<8x32x128xi16>) outs(%[[INIT1]] : tensor<32x8x128xi16>) permutation = [1, 0, 2]
//  CHECK-DAG:  %[[INIT2:.+]] = tensor.empty() : tensor<32x128x11008xi4>
//  CHECK-DAG:  %[[TRANSPOSE1:.+]] = linalg.transpose ins(%[[ARG1]] : tensor<11008x32x128xi4>) outs(%[[INIT2]] : tensor<32x128x11008xi4>) permutation = [1, 2, 0]
//      CHECK:  %[[INIT_EXTSI:.+]] = tensor.empty() : tensor<32x8x128xi32>
//      CHECK:  %[[EXTSI:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP0]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[TRANSPOSE0]] : tensor<32x8x128xi16>) outs(%[[INIT_EXTSI]] : tensor<32x8x128xi32>) {
// CHECK-NEXT:     ^bb0(%[[EXTSI_ARG_IN:.+]]: i16, %[[EXTSI_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[EXTSI_OP:.+]] = arith.extsi %[[EXTSI_ARG_IN]] : i16 to i32
// CHECK-NEXT:     linalg.yield %[[EXTSI_OP]] : i32
//      CHECK:  %[[INIT_EXTUI:.+]] = tensor.empty() : tensor<32x128x11008xi32>
//      CHECK:  %[[EXTUI:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP0]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[TRANSPOSE1]] : tensor<32x128x11008xi4>) outs(%[[INIT_EXTUI]] : tensor<32x128x11008xi32>) {
// CHECK-NEXT:     ^bb0(%[[EXTUI_ARG_IN:.+]]: i4, %[[EXTUI_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[EXTUI_OP:.+]] = arith.extui %[[EXTUI_ARG_IN]] : i4 to i32
// CHECK-NEXT:     linalg.yield %[[EXTUI_OP]] : i32
//      CHECK:  %[[INIT0:.+]] = tensor.empty() : tensor<32x8x11008xi32>
//      CHECK:  %[[FILL:.+]] = linalg.fill ins(%[[CST]] : i32) outs(%[[INIT0]] : tensor<32x8x11008xi32>)
//      CHECK:  %[[MATMUL:.+]] = linalg.batch_matmul ins(%[[EXTSI]], %[[EXTUI]] : tensor<32x8x128xi32>, tensor<32x128x11008xi32>) outs(%[[FILL]] : tensor<32x8x11008xi32>)
//      CHECK:  %[[INIT3:.+]] = tensor.empty() : tensor<11008x32x8xi32>
//      CHECK:  %[[TRANSPOSE2:.+]] = linalg.transpose ins(%[[MATMUL]] : tensor<32x8x11008xi32>) outs(%[[INIT3]] : tensor<11008x32x8xi32>) permutation = [2, 0, 1]
//      CHECK:  return %[[TRANSPOSE2]]

// -----

module {
  func.func @raise_batch_matmul_dyn(%arg0: tensor<8x?x128xi16>, %arg1: tensor<11008x?x128xi4>) -> tensor<11008x?x8xi32> {
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c1 : tensor<8x?x128xi16>
    %0 = tensor.empty(%dim) : tensor<11008x?x8xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<11008x?x8xi32>) -> tensor<11008x?x8xi32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>, 
                                          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, 
                                          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>], 
                         iterator_types = ["parallel", "parallel", "reduction", "parallel"]} 
                         ins(%arg0, %arg1 : tensor<8x?x128xi16>, tensor<11008x?x128xi4>) 
                         outs(%1 : tensor<11008x?x8xi32>) {
    ^bb0(%in: i16, %in_0: i4, %out: i32):
      %3 = arith.extsi %in : i16 to i32
      %4 = arith.extui %in_0 : i4 to i32
      %5 = arith.muli %3, %4 : i32
      %6 = arith.addi %5, %out : i32
      linalg.yield %6 : i32
    } -> tensor<11008x?x8xi32>
    return %2 : tensor<11008x?x8xi32>
  }
}
//  CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//      CHECK:  func @raise_batch_matmul_dyn(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<8x?x128xi16>, %[[ARG1:.+]]: tensor<11008x?x128xi4>
//  CHECK-DAG:  %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:  %[[CST:.+]] = arith.constant 0 : i32
//  CHECK-DAG:  %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<8x?x128xi16>
//  CHECK-DAG:  %[[INIT1:.+]] = tensor.empty(%[[DIM0]]) : tensor<?x8x128xi16>
//  CHECK-DAG:  %[[TRANSPOSE0:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<8x?x128xi16>) outs(%[[INIT1]] : tensor<?x8x128xi16>) permutation = [1, 0, 2]
//  CHECK-DAG:  %[[DIM1:.+]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<11008x?x128xi4>
//  CHECK-DAG:  %[[INIT2:.+]] = tensor.empty(%[[DIM1]]) : tensor<?x128x11008xi4>
//  CHECK-DAG:  %[[TRANSPOSE1:.+]] = linalg.transpose ins(%[[ARG1]] : tensor<11008x?x128xi4>) outs(%[[INIT2]] : tensor<?x128x11008xi4>) permutation = [1, 2, 0]
//      CHECK:  %[[INIT_EXTSI:.+]] = tensor.empty(%[[DIM0]]) : tensor<?x8x128xi32>
//      CHECK:  %[[EXTSI:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP0]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[TRANSPOSE0]] : tensor<?x8x128xi16>) outs(%[[INIT_EXTSI]] : tensor<?x8x128xi32>) {
// CHECK-NEXT:     ^bb0(%[[EXTSI_ARG_IN:.+]]: i16, %[[EXTSI_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[EXTSI_OP:.+]] = arith.extsi %[[EXTSI_ARG_IN]] : i16 to i32
// CHECK-NEXT:     linalg.yield %[[EXTSI_OP]] : i32
//      CHECK:  %[[INIT_EXTUI:.+]] = tensor.empty(%[[DIM1]]) : tensor<?x128x11008xi32>
//      CHECK:  %[[EXTUI:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP0]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[TRANSPOSE1]] : tensor<?x128x11008xi4>) outs(%[[INIT_EXTUI]] : tensor<?x128x11008xi32>) {
// CHECK-NEXT:     ^bb0(%[[EXTUI_ARG_IN:.+]]: i4, %[[EXTUI_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[EXTUI_OP:.+]] = arith.extui %[[EXTUI_ARG_IN]] : i4 to i32
// CHECK-NEXT:     linalg.yield %[[EXTUI_OP]] : i32
//      CHECK:  %[[INIT0:.+]] = tensor.empty(%[[DIM0]]) : tensor<?x8x11008xi32>
//      CHECK:  %[[FILL:.+]] = linalg.fill ins(%[[CST]] : i32) outs(%[[INIT0]] : tensor<?x8x11008xi32>)
//      CHECK:  %[[MATMUL:.+]] = linalg.batch_matmul ins(%[[EXTSI]], %[[EXTUI]] : tensor<?x8x128xi32>, tensor<?x128x11008xi32>) outs(%[[FILL]] : tensor<?x8x11008xi32>)
//      CHECK:  %[[INIT3:.+]] = tensor.empty(%[[DIM0]]) : tensor<11008x?x8xi32>
//      CHECK:  %[[TRANSPOSE2:.+]] = linalg.transpose ins(%[[MATMUL]] : tensor<?x8x11008xi32>) outs(%[[INIT3]] : tensor<11008x?x8xi32>) permutation = [2, 0, 1]
//      CHECK:  return %[[TRANSPOSE2]]
