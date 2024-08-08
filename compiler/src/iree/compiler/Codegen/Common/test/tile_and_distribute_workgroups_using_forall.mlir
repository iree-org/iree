// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups-using-forall-op))" --mlir-print-local-scope --split-input-file %s | FileCheck %s

func.func @matmul_tensors(%0 : tensor<?x?xf32>, %1 : tensor<?x?xf32>, %2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %3 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0]]>}
      ins(%0, %1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_tensors(
//  CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
//  CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
//  CHECK-SAME:     %[[INIT:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//       CHECK:   %[[RESULT:.+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]])
//  CHECK-SAME:       shared_outs(%[[OUTS:.+]] = %[[INIT]])
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][%[[IV0]], 0]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, %[[IV1]]
//   CHECK-DAG:     %[[INIT_SLICE:.+]] = tensor.extract_slice %[[OUTS]][%[[IV0]], %[[IV1]]{{\]}}
//       CHECK:     %[[MATMUL_SLICE:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[LHS_SLICE]], %[[RHS_SLICE]] :
//  CHECK-SAME:         outs(%[[INIT_SLICE]] :
//       CHECK:     scf.forall.in_parallel
//       CHECK:       tensor.parallel_insert_slice %[[MATMUL_SLICE]] into %[[OUTS]][%[[IV0]], %[[IV1]]]
//       CHECK:     mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]
//       CHECK:   return %[[RESULT]]

// -----

func.func @add4D(%0 : index, %1 : index, %2 : index, %3 : index,
    %7 : tensor<?x?x?x?xf32>, %8 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %9 = tensor.empty(%0, %1, %2, %3) : tensor<?x?x?x?xf32>
  %10 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%7, %8 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%9 : tensor<?x?x?x?xf32>)
      attrs = {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 64, 64, 64]]>} {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %11 = arith.addf %arg0, %arg1 : f32
      linalg.yield %11 : f32
    } -> tensor<?x?x?x?xf32>
  return %10 : tensor<?x?x?x?xf32>
}
// CHECK: func.func @add4D(
// CHECK:   %[[RESULT:.+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]], %[[IV2:[a-zA-Z0-9]+]]) =
// CHECK:     %[[TILED_GENERIC:.+]] = linalg.generic
// CHECK:     scf.forall.in_parallel {
// CHECK:       tensor.parallel_insert_slice %[[TILED_GENERIC]] into %{{.+}}[0, %[[IV0]], %[[IV1]], %[[IV2]]{{\]}}
// CHECK:   mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]
//      CHECK:   return %[[RESULT]]

// -----

func.func @add_distribute4D(%0 : index, %1 : index, %2 : index, %3 : index,
  %7 : tensor<?x?x?x?xf32>, %8 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %9 = tensor.empty(%0, %1, %2, %3) : tensor<?x?x?x?xf32>
  %10 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, 
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%7, %8 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%9 : tensor<?x?x?x?xf32>)
      attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[2, 64, 64, 64]]>} {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %11 = arith.addf %arg0, %arg1 : f32
      linalg.yield %11 : f32
    } -> tensor<?x?x?x?xf32>
  return %10 : tensor<?x?x?x?xf32>
}
//      CHECK: func.func @add_distribute4D(
//      CHECK:   %[[RESULT:.+]] = scf.forall
// CHECK-SAME:       (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]], %[[IV2:[a-zA-Z0-9]+]], %[[IV3:[a-zA-Z0-9]+]]) =
//      CHECK:     %[[TILED_GENERIC:.+]] = linalg.generic
//      CHECK:     scf.forall.in_parallel {
//      CHECK:       tensor.parallel_insert_slice %[[TILED_GENERIC]] into %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]{{\]}}
//      CHECK:   mapping = [#iree_codegen.workgroup_mapping<z:1>, #iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]
//      CHECK:   return %[[RESULT]]
