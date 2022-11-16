// RUN: iree-opt --pass-pipeline="builtin.module(func.func(linalg-fuse{tiling-level=0 vectorize}), canonicalize, cse)" --split-input-file %s | FileCheck %s

func.func @matmul_bias_add(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[10, 20, 30]]>}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1)-> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%1, %arg2 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%init : tensor<?x?xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
    } -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
//      CHECK: func.func @matmul_bias_add(
//      CHECK:   scf.for
// CHECK-SAME:   {
//      CHECK:     scf.for
// CHECK-SAME:     {
//      CHECK:       linalg.fill
//      CHECK:       linalg.matmul
//      CHECK:       linalg.generic
//      CHECK:     }
//      CHECK:   }

// -----

func.func @matmul_bias_add_static(%arg0 : tensor<20x60xf32>, %arg1 : tensor<60x120xf32>, %arg2 : tensor<120xf32>) -> tensor<20x120xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = tensor.empty() : tensor<20x120xf32>
  %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<20x120xf32>) -> tensor<20x120xf32>
  %1 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[10, 20, 30]]>}
      ins(%arg0, %arg1 : tensor<20x60xf32>, tensor<60x120xf32>)
      outs(%0 : tensor<20x120xf32>) -> tensor<20x120xf32>
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1)-> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%1, %arg2 : tensor<20x120xf32>, tensor<120xf32>)
    outs(%init : tensor<20x120xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
    } -> tensor<20x120xf32>
  return %2 : tensor<20x120xf32>
}
//      CHECK: func.func @matmul_bias_add_static(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<20x60xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<60x120xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<120xf32>
//  CHECK-DAG:   %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<10x20xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C10:.+]] = arith.constant 10 : index
//  CHECK-DAG:   %[[C20:.+]] = arith.constant 20 : index
//  CHECK-DAG:   %[[C60:.+]] = arith.constant 60 : index
//  CHECK-DAG:   %[[C120:.+]] = arith.constant 120 : index
//      CHECK:   %[[INIT:.+]] = tensor.empty() : tensor<20x120xf32>
//      CHECK:   %[[RESULT:.+]] = scf.for %[[IV0:.+]] = %[[C0]] to %[[C20]] step %[[C10]]
// CHECK-SAME:       iter_args(%[[ARG4:.+]] = %[[INIT]])
//      CHECK:     %[[YIELD:.+]] = scf.for %[[IV1:.+]] = %[[C0]] to %[[C120]]
// CHECK-SAME:         iter_args(%[[ARG6:.+]] = %[[ARG4]])
//      CHECK:       %[[REDUCTION:.+]] = scf.for %[[IV2:.+]] = %[[C0]] to %[[C60]]
// CHECK-SAME:           iter_args(%[[ARG7:.+]] = %[[CST]])
//  CHECK-DAG:         %[[LHS:.+]] = vector.transfer_read %[[ARG0]][%[[IV0]], %[[IV2]]]
//  CHECK-DAG:         %[[RHS:.+]] = vector.transfer_read %[[ARG1]][%[[IV2]], %[[IV1]]]
//      CHECK:         %[[CONTRACT:.+]] = vector.contract
// CHECK-SAME:             kind = #vector.kind<add>
// CHECK-SAME:             %[[LHS]], %[[RHS]], %[[ARG7]]
//      CHECK:         scf.yield %[[CONTRACT]]
//  CHECK-DAG:       %[[BIAS:.+]] = vector.transfer_read %[[ARG2]][%[[IV1]]]
//      CHECK:       %[[BROADCAST_BIAS:.+]] = vector.broadcast %[[BIAS]]
//      CHECK:       %[[ADD:.+]] = arith.addf %[[REDUCTION]], %[[BROADCAST_BIAS]]
//      CHECK:       %[[INSERT:.+]] = vector.transfer_write %[[ADD]], %[[ARG6]][%[[IV0]], %[[IV1]]]
//      CHECK:       scf.yield %[[INSERT]]
//      CHECK:     scf.yield %[[YIELD]]
//      CHECK:   return %[[RESULT]]
