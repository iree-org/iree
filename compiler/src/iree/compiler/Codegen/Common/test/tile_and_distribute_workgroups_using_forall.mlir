// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups-using-forall-op, cse))" --mlir-print-local-scope --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups-using-forall-op{transpose-workgroup=true}, cse))" --mlir-print-local-scope --split-input-file %s | FileCheck %s --check-prefix=TRANSPOSE

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
// CHECK-LABEL: func.func @add4D(
//       CHECK:   %[[RESULT:.+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]], %[[IV2:[a-zA-Z0-9]+]]) =
//       CHECK:     %[[TILED_GENERIC:.+]] = linalg.generic
//       CHECK:     scf.forall.in_parallel {
//       CHECK:       tensor.parallel_insert_slice %[[TILED_GENERIC]] into %{{.+}}[0, %[[IV0]], %[[IV1]], %[[IV2]]{{\]}}
//       CHECK:   mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]
//       CHECK:   return %[[RESULT]]

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
// CHECK-LABEL: func.func @add_distribute4D(
//       CHECK:   %[[RESULT:.+]] = scf.forall
//  CHECK-SAME:       (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]], %[[IV2:[a-zA-Z0-9]+]], %[[IV3:[a-zA-Z0-9]+]]) =
//       CHECK:     %[[TILED_GENERIC:.+]] = linalg.generic
//       CHECK:     scf.forall.in_parallel {
//       CHECK:       tensor.parallel_insert_slice %[[TILED_GENERIC]] into %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]{{\]}}
//       CHECK:   mapping = [#iree_codegen.workgroup_mapping<z:1>, #iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]
//       CHECK:   return %[[RESULT]]

// -----

func.func @add_distribute4D_zero_tile_size(%0 : index, %1 : index, %2 : index, %3 : index,
  %7 : tensor<?x?x?x?xf32>, %8 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %9 = tensor.empty(%0, %1, %2, %3) : tensor<?x?x?x?xf32>
  %10 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%7, %8 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%9 : tensor<?x?x?x?xf32>)
      attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[2, 64, 0, 64]]>} {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %11 = arith.addf %arg0, %arg1 : f32
      linalg.yield %11 : f32
    } -> tensor<?x?x?x?xf32>
  return %10 : tensor<?x?x?x?xf32>
}
// CHECK-LABEL: func.func @add_distribute4D_zero_tile_size(
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %{{.+}}, %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %{{.+}}, %[[C1]]
//   CHECK-DAG:   %[[D2:.+]] = tensor.dim %{{.+}}, %[[C2]]
//   CHECK-DAG:   %[[D3:.+]] = tensor.dim %{{.+}}, %[[C3]]
//       CHECK:   %[[RESULT:.+]] = scf.forall
//  CHECK-SAME:       (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]], %[[IV2:[a-zA-Z0-9]+]]) =
//  CHECK-SAME:       to (%[[D0]], %[[D1]], %[[D3]])
//       CHECK:     %[[SLICE:.+]] = tensor.extract_slice %{{.+}}[%[[IV0]], %[[IV1]], 0, %[[IV2]]] [%{{.+}}, %{{.+}}, %[[D2]], %{{.+}}]
//       CHECK:     %[[TILED_GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[SLICE]],
//       CHECK:     scf.forall.in_parallel {
//       CHECK:       tensor.parallel_insert_slice %[[TILED_GENERIC]] into %{{.+}}[%[[IV0]], %[[IV1]], 0, %[[IV2]]{{\]}}
//       CHECK:   mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]
//       CHECK:   return %[[RESULT]]

// -----

func.func @gemm_unit_N(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x1xf32>,
    %arg2 : tensor<?x1xf32>) -> tensor<?x1xf32> {
  %0 = linalg.matmul {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 64]]>}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x1xf32>)
      outs(%arg2 : tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}
// CHECK-LABEL: func.func @gemm_unit_N(
//       CHECK:   %[[RESULT:.+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]])
//       CHECK:     %[[MATMUL:.+]] = linalg.matmul
//       CHECK:     scf.forall.in_parallel {
//       CHECK:       tensor.parallel_insert_slice %[[MATMUL]] into %{{.+}}[%[[IV0]], 0] [%{{.+}}, 1]

// -----

func.func @generic_unit_dims(%arg0 : tensor<1x?x1x1x?x?x1x?xf32>) -> tensor<1x?x1x1x?x?x1x?xf32> {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %d1 = tensor.dim %arg0, %c1 : tensor<1x?x1x1x?x?x1x?xf32>
  %d4 = tensor.dim %arg0, %c4 : tensor<1x?x1x1x?x?x1x?xf32>
  %d5 = tensor.dim %arg0, %c5 : tensor<1x?x1x1x?x?x1x?xf32>
  %d7 = tensor.dim %arg0, %c7 : tensor<1x?x1x1x?x?x1x?xf32>
  %empty = tensor.empty(%d1, %d4, %d5, %d7) : tensor<1x?x1x1x?x?x1x?xf32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<1x?x1x1x?x?x1x?xf32>) outs(%empty : tensor<1x?x1x1x?x?x1x?xf32>)
      attrs = {lowering_config = #iree_codegen.lowering_config<tile_sizes=[[0, 0, 0, 0, 64, 64, 0, 64]]>} {
    ^bb0(%b0: f32, %b1: f32):
      %9 = arith.addf %b0, %b0 : f32
      linalg.yield %9 : f32
  } -> tensor<1x?x1x1x?x?x1x?xf32>
  return %0 : tensor<1x?x1x1x?x?x1x?xf32>
}
// CHECK-LABEL: func.func @generic_unit_dims(
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
//   CHECK-DAG:   %[[C7:.+]] = arith.constant 7 : index
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %{{.+}}, %[[C1]]
//   CHECK-DAG:   %[[D4:.+]] = tensor.dim %{{.+}}, %[[C4]]
//   CHECK-DAG:   %[[D5:.+]] = tensor.dim %{{.+}}, %[[C5]]
//   CHECK-DAG:   %[[D7:.+]] = tensor.dim %{{.+}}, %[[C7]]
//       CHECK:   %[[RESULT:.+]] = scf.forall
//  CHECK-SAME:       (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]], %[[IV2:[a-zA-Z0-9]+]]) =
//  CHECK-SAME:       to (%[[D4]], %[[D5]], %[[D7]])
//       CHECK:     %[[SLICE:.+]] = tensor.extract_slice %{{.+}}[0, 0, 0, 0, %[[IV0]], %[[IV1]], 0, %[[IV2]]]
//  CHECK-SAME:         [1, %[[D1]], 1, 1, %{{.+}}, %{{.+}}, 1, %{{.+}}]
//       CHECK:     %[[TILED_GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[SLICE]] :
//       CHECK:     scf.forall.in_parallel {
//       CHECK:       tensor.parallel_insert_slice %[[TILED_GENERIC]] into %{{.+}}[0, 0, 0, 0, %[[IV0]], %[[IV1]], 0, %[[IV2]]{{\]}}
//  CHECK-SAME:           [1, %[[D1]], 1, 1, %{{.+}}, %{{.+}}, 1, %{{.+}}]
//       CHECK:   mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]
//       CHECK:   return %[[RESULT]]

// -----

func.func @reduce_to_scalar(%arg0 : tensor<?xf32>, %arg1 : tensor<f32>) -> tensor<f32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
      ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<f32>)
      attrs = {lowering_config = #iree_codegen.lowering_config<tile_sizes=[[]]>} {
    ^bb0(%b0 : f32, %b1 : f32) :
      %1 = arith.addf %b0, %b1 : f32
      linalg.yield %1 : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}
// CHECK-LABEL: func @reduce_to_scalar(
//   CHECK-NOT:   scf.forall

// -----

func.func @scalar(%arg0 : tensor<f32>) -> tensor<f32> {
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []}
      ins(%arg0 : tensor<f32>) outs(%0 : tensor<f32>)
      attrs = {lowering_config = #iree_codegen.lowering_config<tile_sizes=[[]]>} {
    ^bb0(%b0 : f32, %b1 : f32) :
      %1 = arith.addf %b0, %b0 : f32
      linalg.yield %1 : f32
  } -> tensor<f32>
  return %1 : tensor<f32>
}
// CHECK-LABEL: func @scalar(
//   CHECK-NOT:   scf.forall

// -----

func.func @matmul_interchange(%0 : tensor<?x?xf32>,
    %1 : tensor<?x?xf32>, %2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %3 = linalg.matmul
      {lowering_config = #iree_codegen.lowering_config<tile_sizes = [{sizes = [32, 64, 0], interchange = [1, 0, 2]}]>}
      ins(%0, %1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_interchange(
//  CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
//  CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
//  CHECK-SAME:     %[[INIT:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//   CHECK-DAG:   %[[M:.+]] = tensor.dim %[[LHS]], %[[C0]]
//   CHECK-DAG:   %[[N:.+]] = tensor.dim %[[RHS]], %[[C1]]
//       CHECK:   %[[RESULT:.+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]])
//  CHECK-SAME:       to (%[[N]], %[[M]]) step (64, 32)
//  CHECK-SAME:       shared_outs(%[[OUTS:.+]] = %[[INIT]])
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][%[[IV1]], 0]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, %[[IV0]]
//   CHECK-DAG:     %[[INIT_SLICE:.+]] = tensor.extract_slice %[[OUTS]][%[[IV1]], %[[IV0]]{{\]}}
//       CHECK:     %[[MATMUL_SLICE:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[LHS_SLICE]], %[[RHS_SLICE]] :
//  CHECK-SAME:         outs(%[[INIT_SLICE]] :
//       CHECK:     scf.forall.in_parallel
//       CHECK:       tensor.parallel_insert_slice %[[MATMUL_SLICE]] into %[[OUTS]][%[[IV1]], %[[IV0]]]
//       CHECK:     mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]
//       CHECK:   return %[[RESULT]]

// -----

func.func @no_compute(%arg0 : memref<?x?x?xf32>, %arg1 : memref<?x?x?xf32>) {
  util.optimization_barrier %arg0 : memref<?x?x?xf32>
  util.optimization_barrier %arg1 : memref<?x?x?xf32>
  return
}
// CHECK-LABEL: @no_compute(
//   CHECK-NOT:   scf.forall

// -----

func.func @matmul_memrefs(%0 : memref<?x?xf32>, %1 : memref<?x?xf32>, %2 : memref<?x?xf32>) {
  linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0]]>}
      ins(%0, %1 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%2 : memref<?x?xf32>)
  return
}
// CHECK-LABEL: func @matmul_memrefs(
//       CHECK:   scf.forall
//       CHECK:     linalg.matmul
//       CHECK:   mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]

// -----

func.func @matmul_fusion_test(%arg0 : tensor<?x?xf16>,
    %arg1 : tensor<?x?xf16>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst0 = arith.constant 0.0 : f32
  %M = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %N = tensor.dim %arg1, %c1 : tensor<?x?xf16>
  %K = tensor.dim %arg0, %c1 : tensor<?x?xf16>
  %empty_lhs = tensor.empty(%M, %K) : tensor<?x?xf32>
  %extf_lhs = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?x?xf16>) outs(%empty_lhs : tensor<?x?xf32>) {
    ^bb0(%b0 : f16, %b1 : f32) :
      %0 = arith.extf %b0 : f16 to f32
      linalg.yield %0 : f32
  } -> tensor<?x?xf32>
  %empty_rhs = tensor.empty(%K, %N) : tensor<?x?xf32>
  %extf_rhs = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : tensor<?x?xf16>) outs(%empty_rhs : tensor<?x?xf32>) {
    ^bb0(%b0 : f16, %b1 : f32) :
      %0 = arith.extf %b0 : f16 to f32
      linalg.yield %0 : f32
  } -> tensor<?x?xf32>
  %empty = tensor.empty(%M, %N) : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst0 : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  %matmul = linalg.matmul
      {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>}
      ins(%extf_lhs, %extf_rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %matmul : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_fusion_test
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf16>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf16>
//       CHECK:   %[[RESULT:.+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]]) =
//       CHECK:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], 0]
//       CHECK:     %[[LHS:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[LHS_SLICE]] :
//       CHECK:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[IV1]]]
//       CHECK:     %[[RHS:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[RHS_SLICE]] :
//       CHECK:     %[[FILL:.+]] = linalg.fill
//       CHECK:     %[[MATMUL:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:         outs(%[[FILL]] :
//       CHECK:     scf.forall.in_parallel
//       CHECK:       tensor.parallel_insert_slice %[[MATMUL]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @avoid_unit_range_distribute(
    %arg0 : tensor<32x?x?x16x16xf16>, %arg1 : tensor<32x?x8x16x16xf16>,
    %arg2 : tensor<32x?x16x8x16xf16>) -> tensor<32x?x16x8x16xf16> {
   %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d5, d2, d6)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, d3, d6, d4)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
      ins(%arg0, %arg1 : tensor<32x?x?x16x16xf16>, tensor<32x?x8x16x16xf16>)
      outs(%arg2 : tensor<32x?x16x8x16xf16>)
      attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 16, 1, 16, 1, 16]]>} {
    ^bb0(%b0: f16, %b1: f16, %b2 : f16):
      %1 = arith.mulf %b0, %b1 : f16
      %2 = arith.addf %b2, %1 : f16
      linalg.yield %2 : f16
  } -> tensor<32x?x16x8x16xf16>
  return %0 : tensor<32x?x16x8x16xf16>
}
// CHECK-LABEL: func @avoid_unit_range_distribute(
//       CHECK:   scf.forall (%{{[a-zA-Z0-9]+}}, %{{[a-zA-Z0-9]+}}, %{{[a-zA-Z0-9]+}}) in (32, %{{.+}}, 8)
//       CHECK:   mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]

// -----

// This just verifies that constant dim propagation works as expected after tiling.
func.func @set_size_to_tilesize_when_divisible(
    %arg0 : tensor<?x16x32x128xf16>, %arg1 : tensor<4096x32x128xf16>,
    %arg2 : tensor<?x16x4096xf16>) -> tensor<?x16x4096xf16> {
   %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
      ins(%arg0, %arg1 : tensor<?x16x32x128xf16>, tensor<4096x32x128xf16>)
      outs(%arg2 : tensor<?x16x4096xf16>)
      attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 16, 128, 1, 128]]>} {
    ^bb0(%b0: f16, %b1: f16, %b2 : f16):
      %1 = arith.mulf %b0, %b1 : f16
      %2 = arith.addf %b2, %1 : f16
      linalg.yield %2 : f16
  } -> tensor<?x16x4096xf16>
  return %0 : tensor<?x16x4096xf16>
}
// CHECK-LABEL: func @set_size_to_tilesize_when_divisible(
//       CHECK:   scf.forall
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//       CHECK:     scf.forall.in_parallel
//       CHECK:       tensor.parallel_insert_slice %[[GENERIC]]
//  CHECK-SAME:           tensor<1x16x128xf16> into tensor<?x16x4096xf16>

// -----

func.func @matmul_consumer_fusion_test(%arg0 : tensor<?x?xf16>,
    %arg1 : tensor<?x?xf16>, %arg2: tensor<?xf16>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst0 = arith.constant 0.0 : f32
  %M = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %N = tensor.dim %arg1, %c1 : tensor<?x?xf16>
  %K = tensor.dim %arg0, %c1 : tensor<?x?xf16>
  %empty_lhs = tensor.empty(%M, %K) : tensor<?x?xf32>
  %extf_lhs = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?x?xf16>) outs(%empty_lhs : tensor<?x?xf32>) {
    ^bb0(%b0 : f16, %b1 : f32) :
      %0 = arith.extf %b0 : f16 to f32
      linalg.yield %0 : f32
  } -> tensor<?x?xf32>
  %empty_rhs = tensor.empty(%K, %N) : tensor<?x?xf32>
  %extf_rhs = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : tensor<?x?xf16>) outs(%empty_rhs : tensor<?x?xf32>) {
    ^bb0(%b0 : f16, %b1 : f32) :
      %0 = arith.extf %b0 : f16 to f32
      linalg.yield %0 : f32
  } -> tensor<?x?xf32>
  %empty = tensor.empty(%M, %N) : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst0 : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  %matmul = linalg.matmul
      {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>}
      ins(%extf_lhs, %extf_rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  %empty_biasadd = tensor.empty(%M, %N) : tensor<?x?xf32>
  %bias_add = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
    ins(%matmul,%arg2 : tensor<?x?xf32>, tensor<?xf16>) outs(%empty_biasadd : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1: f16, %b2 : f32) :
      %0 = arith.extf %b1 : f16 to f32
      %1 = arith.addf %b0, %0 : f32
      linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  %empty_relu = tensor.empty(%M, %N) : tensor<?x?xf32>
  %relu = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
    ins(%bias_add : tensor<?x?xf32>) outs(%empty_relu : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32) :
      %0 = arith.maximumf %b0, %cst0 : f32
      linalg.yield %0 : f32
  } -> tensor<?x?xf32>
  return %relu : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_consumer_fusion_test(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf16>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf16>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?xf16>
//       CHECK:   %[[RESULT:.+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]]) =
//       CHECK:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], 0]
//       CHECK:     %[[LHS:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[LHS_SLICE]] :
//       CHECK:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[IV1]]]
//       CHECK:     %[[RHS:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[RHS_SLICE]] :
//       CHECK:     %[[FILL:.+]] = linalg.fill
//       CHECK:     %[[MATMUL:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:         outs(%[[FILL]] :
//       CHECK:     %[[BIASADD:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[MATMUL]]
//       CHECK:     %[[RELU:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[BIASADD]] :
//       CHECK:     scf.forall.in_parallel
//       CHECK:       tensor.parallel_insert_slice %[[RELU]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @multi_result(%arg0: tensor<64x128xf32>, %arg1: tensor<128x256xf32>, %arg2: tensor<256xf32>) -> (tensor<64x256xf32>, tensor<64x256xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<64x256xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x256xf32>) -> tensor<64x256xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<128x256xf32>) outs(%1 : tensor<64x256xf32>) -> tensor<64x256xf32>
  %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%2, %arg2 : tensor<64x256xf32>, tensor<256xf32>)
      outs(%0 : tensor<64x256xf32>)
      attrs = {lowering_config =
        #iree_codegen.lowering_config<tile_sizes = [[16, 64]]>} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.addf %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<64x256xf32>
  return %2, %3 : tensor<64x256xf32>, tensor<64x256xf32>
}

// CHECK-LABEL: func @multi_result(
//       CHECK:   %[[RESULT:.+]]:2 = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]])
//  CHECK-SAME:       shared_outs(%[[OUTS:.+]] = {{.*}}, %[[OUTS:.+]] = {{.*}})
//       CHECK:      linalg.matmul
//       CHECK:      linalg.generic
//       CHECK:     scf.forall.in_parallel
//       CHECK:       tensor.parallel_insert_slice
//       CHECK:       tensor.parallel_insert_slice
//       CHECK:  return %[[RESULT]]#1, %[[RESULT]]#0

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @multi_use_producer_no_yield_replacement(%7: tensor<12x197x197xf32>) -> tensor<12x197x197xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant -3.40282347E+38 : f32
  %8 = tensor.empty() : tensor<12x197x197xf32>
  %9 = tensor.empty() : tensor<12x197xf32>
  %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<12x197xf32>) -> tensor<12x197xf32>
  %11 = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%7 : tensor<12x197x197xf32>) outs(%10 : tensor<12x197xf32>) {
  ^bb0(%in: f32, %out: f32):
    %15 = arith.maxnumf %in, %out : f32
    linalg.yield %15 : f32
  } -> tensor<12x197xf32>
  %12 = linalg.fill ins(%cst : f32) outs(%9 : tensor<12x197xf32>) -> tensor<12x197xf32>
  %13 = linalg.generic {
    indexing_maps = [#map, #map1, #map1],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%7, %11 : tensor<12x197x197xf32>, tensor<12x197xf32>)
    outs(%12 : tensor<12x197xf32>) attrs =  {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[4, 8, 0]]>} {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %15 = arith.subf %in, %in_1 : f32
    %16 = math.exp %15 : f32
    %17 = arith.addf %16, %out : f32
    linalg.yield %17 : f32
  } -> tensor<12x197xf32>
  %14:2 = linalg.generic {
    indexing_maps = [#map, #map1, #map1, #map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%7, %11, %13 : tensor<12x197x197xf32>, tensor<12x197xf32>, tensor<12x197xf32>)
    outs(%8, %8 : tensor<12x197x197xf32>, tensor<12x197x197xf32>) {
  ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32, %out_3: f32):
    %15 = arith.subf %in, %in_1 : f32
    %16 = math.exp %15 : f32
    %17 = arith.divf %16, %in_2 : f32
    linalg.yield %16, %17 : f32, f32
  } -> (tensor<12x197x197xf32>, tensor<12x197x197xf32>)
  return %14#1 : tensor<12x197x197xf32>
}

// CHECK-LABEL: func @multi_use_producer_no_yield_replacement(
//       CHECK:   %[[RESULT:.+]] = scf.forall
//       CHECK:     %[[MAX:.+]] = linalg.generic
//       CHECK:       arith.maxnumf
//       CHECK:     %[[EXPSUM:.+]] = linalg.generic
//  CHECK-SAME:       ins(%{{.*}}, %[[MAX]]
//       CHECK:       arith.subf
//       CHECK:       math.exp
//       CHECK:       arith.addf
//       CHECK:     %[[EXPDIV:.+]] = linalg.generic
//  CHECK-SAME:       ins(%{{.*}}, %[[MAX]], %[[EXPSUM]]
//       CHECK:       arith.subf
//       CHECK:       math.exp
//       CHECK:       arith.divf
//       CHECK:   return %[[RESULT]]

// -----

// Fusion of the following graph, root marked with [brackets].
//   A
//  / \
// B  [C]
//  \ /
//   D
#map = affine_map<(d0) -> (d0)>
func.func @diamond_graph(%0: tensor<12xf32>, %1: tensor<12xf32>) -> tensor<12xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %2 = tensor.empty() : tensor<12xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]
  } ins(%0, %1 : tensor<12xf32>, tensor<12xf32>) outs(%2 : tensor<12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %7 = arith.addf %in, %in_0 : f32
    linalg.yield %7 : f32
  } -> tensor<12xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]
  } ins(%3, %0 : tensor<12xf32>, tensor<12xf32>) outs(%2 : tensor<12xf32>) attrs =  {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[4]]>} {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %8 = arith.addf %in, %in_1 : f32
    linalg.yield %8 : f32
  } -> tensor<12xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]
  } ins(%3, %1 : tensor<12xf32>, tensor<12xf32>) outs(%2 : tensor<12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %9 = arith.addf %in, %in_0 : f32
    linalg.yield %9 : f32
  } -> tensor<12xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]
  } ins(%4, %5 : tensor<12xf32>, tensor<12xf32>) outs(%2 : tensor<12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %10 = arith.addf %in, %in_0 : f32
    linalg.yield %10 : f32
  } -> tensor<12xf32>
  return %6 : tensor<12xf32>
}

// CHECK-LABEL: func @diamond_graph(
//       CHECK:   %[[RESULT:.+]] = scf.forall
//       CHECK:     %[[TOP:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[IN0_SLICE:.+]], %[[IN1_SLICE:.+]]
//   CHECK-DAG:     %[[LEFT:.+]] = linalg.generic {{.*}} ins(%[[TOP]], %[[IN0_SLICE]]
//   CHECK-DAG:     %[[RIGHT:.+]] = linalg.generic {{.*}} ins(%[[TOP]], %[[IN1_SLICE]]
//       CHECK:     linalg.generic {{.*}} ins(%[[LEFT]], %[[RIGHT]]
//       CHECK:   return %[[RESULT]]

// -----

// Fusion of the following graph, root marked with [brackets].
// [A] B
//  \ /
//   C
#map = affine_map<(d0) -> (d0)>
func.func @v_shaped_graph(%0: tensor<12xf32>, %1: tensor<12xf32>) -> tensor<12xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %2 = tensor.empty() : tensor<12xf32>
  %A = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]
  } ins(%0 : tensor<12xf32>) outs(%2 : tensor<12xf32>) attrs =  {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[4]]>} {
  ^bb0(%in: f32, %out: f32):
    %6 = math.sqrt %in : f32
    linalg.yield %6 : f32
  } -> tensor<12xf32>
  %B = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]
  } ins(%1 : tensor<12xf32>) outs(%2 : tensor<12xf32>) {
  ^bb0(%in: f32, %out: f32):
    %7 = math.sqrt %in : f32
    linalg.yield %7 : f32
  } -> tensor<12xf32>
  %C = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]
  } ins(%A, %B : tensor<12xf32>, tensor<12xf32>) outs(%2 : tensor<12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %8 = arith.addf %in, %in_0 : f32
    linalg.yield %8 : f32
  } -> tensor<12xf32>
  return %C : tensor<12xf32>
}

// CHECK-LABEL: func @v_shaped_graph(
//  CHECK-SAME:   %[[IN0:[A-Za-z0-9]+]]: tensor<12xf32>
//  CHECK-SAME:   %[[IN1:[A-Za-z0-9]+]]: tensor<12xf32>
//       CHECK:   %[[RESULT:.+]] = scf.forall
//   CHECK-DAG:     %[[SLICE0:.+]] = tensor.extract_slice %[[IN0]]
//   CHECK-DAG:     %[[SLICE1:.+]] = tensor.extract_slice %[[IN1]]
//   CHECK-DAG:     %[[LEFT:.+]] = linalg.generic {{.*}} ins(%[[SLICE0]]
//   CHECK-DAG:     %[[RIGHT:.+]] = linalg.generic {{.*}} ins(%[[SLICE1]]
//       CHECK:     linalg.generic {{.*}} ins(%[[LEFT]], %[[RIGHT]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @consumer_fuse_scatter(%arg0: tensor<3x2048x2048xf32>,
                                 %arg1: tensor<3x2048x2048xf32>,
                                 %arg2: tensor<3x1xi32>) -> tensor<3x2048x2048xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<3x2048x2048xf32>
  %1 = linalg.add {lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 1, 256]}>}
    ins(%arg0, %arg1 : tensor<3x2048x2048xf32>, tensor<3x2048x2048xf32>) outs(%0 : tensor<3x2048x2048xf32>) -> tensor<3x2048x2048xf32>
  %2 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%1, %arg2 : tensor<3x2048x2048xf32>, tensor<3x1xi32>) outs(%0 : tensor<3x2048x2048xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):
    iree_linalg_ext.yield %arg3 : f32
  } -> tensor<3x2048x2048xf32>
  return %2 : tensor<3x2048x2048xf32>
}

// CHECK-LABEL: func @consumer_fuse_scatter(
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<3x2048x2048xf32>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<3x2048x2048xf32>
//  CHECK-SAME:   %[[IND:[A-Za-z0-9]+]]: tensor<3x1xi32>
//       CHECK:   %[[RESULT:.+]] = scf.forall (%[[ID0:.+]], %[[ID1:.+]], %[[ID2:[A-Za-z0-9]+]]) {{.*}} shared_outs(%[[DEST:.+]] = %{{.*}})
//   CHECK-DAG:     %[[SRC:.+]] = linalg.add
//   CHECK-DAG:     %[[IND_SLICE:.+]] = tensor.extract_slice %[[IND]][%[[ID0]], 0] {{.*}} : tensor<3x1xi32> to tensor<1x1xi32>
//   CHECK-DAG:     %[[DEST_SLICE:.+]] = tensor.extract_slice %[[DEST]][0, %[[ID1]], %[[ID2]]] {{.*}} to tensor<3x1x256xf32>
//       CHECK:     %[[SCATTER:.+]] = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
//  CHECK-SAME:       ins(%[[SRC]], %[[IND_SLICE]]{{.*}} outs(%[[DEST_SLICE]]
//       CHECK:       tensor.parallel_insert_slice %[[SCATTER]] into %[[DEST]][0, %[[ID1]], %[[ID2]]]

// -----

func.func @dont_transpose_dynamic(%0 : tensor<?x?xf32>, %1 : tensor<?x?xf32>, %2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %3 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0]]>}
      ins(%0, %1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

// TRANSPOSE-LABEL: func @dont_transpose_dynamic(
//       TRANSPOSE:   scf.forall
//       TRANSPOSE:    [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]

// -----

func.func @transpose_static(%0 : tensor<128x128xf32>, %1 : tensor<128x128xf32>, %2 : tensor<128x128xf32>) -> tensor<128x128xf32> {
  %3 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0]]>}
      ins(%0, %1 : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %3 : tensor<128x128xf32>
}

// TRANSPOSE-LABEL: func @transpose_static(
//       TRANSPOSE:   scf.forall
//       TRANSPOSE:    [#iree_codegen.workgroup_mapping<x>, #iree_codegen.workgroup_mapping<y>]

// -----

func.func @only_transpose_x_y(%7 : tensor<128x128x128x128xf32>, %8 : tensor<128x128x128x128xf32>) -> tensor<128x128x128x128xf32> {
  %9 = tensor.empty() : tensor<128x128x128x128xf32>
  %10 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%7, %8 : tensor<128x128x128x128xf32>, tensor<128x128x128x128xf32>)
      outs(%9 : tensor<128x128x128x128xf32>)
      attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[2, 64, 64, 64]]>} {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %11 = arith.addf %arg0, %arg1 : f32
      linalg.yield %11 : f32
    } -> tensor<128x128x128x128xf32>
  return %10 : tensor<128x128x128x128xf32>
}

// TRANSPOSE-LABEL: func @only_transpose_x_y(
//       TRANSPOSE:   scf.forall
//       TRANSPOSE:     mapping = [#iree_codegen.workgroup_mapping<z:1>, #iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<x>, #iree_codegen.workgroup_mapping<y>]

// -----

// Incase of less than 2 workgroup_mapping, don't apply transpose.
func.func @dont_transpose_less(%0 : tensor<128x128xf32>, %1 : tensor<128x128xf32>, %2 : tensor<128x128xf32>) -> tensor<128x128xf32> {
  %3 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 0, 0]]>}
      ins(%0, %1 : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %3 : tensor<128x128xf32>
}

// TRANSPOSE-LABEL: func @dont_transpose_less(
//       TRANSPOSE:   scf.forall
//       TRANSPOSE:    [#iree_codegen.workgroup_mapping<x>]

// -----

func.func @set_encoding_gpu(%arg0 : tensor<?x?xi8>) -> tensor<?x?x8x4x4x4x2x8xi8> {
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xi8>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xi8>
  %s0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 128)>()[%d0]
  %s1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%d1]
  %22 = tensor.empty(%s0, %s1) : tensor<?x?x128x64xi8>
  %pack = linalg.pack %arg0 padding_value(%c0_i8 : i8)
      outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [128, 64]
      into %22 : tensor<?x?xi8> -> tensor<?x?x128x64xi8>
  %expanded = tensor.expand_shape %pack [[0], [1], [2, 3, 4], [5, 6, 7]]
      output_shape [%s0, %s1, 4, 8, 4, 2, 4, 8]
      : tensor<?x?x128x64xi8> into tensor<?x?x4x8x4x2x4x8xi8>
  %23 = tensor.empty(%s0, %s1) : tensor<?x?x8x4x4x4x2x8xi8>
  %24 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d4, d2, d5, d6, d3, d7)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded : tensor<?x?x4x8x4x2x4x8xi8>) outs(%23 : tensor<?x?x8x4x4x4x2x8xi8>)
      attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 8, 4, 4, 4, 2, 8]]>} {
  ^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
  } -> tensor<?x?x8x4x4x4x2x8xi8>
  return %24 : tensor<?x?x8x4x4x4x2x8xi8>
}

// CHECK-LABEL: func @set_encoding_gpu(
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<?x?xi8>
//       CHECK:   %[[RESULT:.+]] = scf.forall (%[[ID0:.+]], %[[ID1:.+]])
//       CHECK:     linalg.pack
//       CHECK:     tensor.expand_shape
//       CHECK:     linalg.generic
//       CHECK:     tensor.parallel_insert_slice

// -----

func.func @pad_fusion(%0 : tensor<?x?xf32>, %1 : tensor<?x?xf32>, %2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %padded = tensor.pad %0 low[1, 1] high[1, 1] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  %3 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0]]>}
      ins(%padded, %1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

// CHECK-LABEL: func @pad_fusion(
//       CHECK: %[[RESULT:.+]] = scf.forall (%[[ID0:.+]], %[[ID1:.+]])
//       CHECK:   %[[PADDED:.+]] = tensor.pad
//       CHECK:   %[[MATMUL:.+]] = linalg.matmul
//  CHECK-SAME:   ins(%[[PADDED]]

// -----

// Test 1 of 2 that are testing fusion while considering multiple slices.

func.func @horizontal_fusion_consumer_fusion1(%arg0 : tensor<2x4096x640xf16>,
    %arg1 : tensor<10x64x640xf16>, %arg2 : tensor<10x64x640xf16>, %arg3 : tensor<10x64x640xf16>)
    -> (tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>) {
  %cst = arith.constant 0.0 : f32
  %11 = tensor.empty() : tensor<2x10x4096x64xf16>
  %12 = tensor.empty() : tensor<2x10x4096x64xf32>
  %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<2x10x4096x64xf32>) -> tensor<2x10x4096x64xf32>
  %14:3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1, %arg2, %arg3
          : tensor<2x4096x640xf16>, tensor<10x64x640xf16>, tensor<10x64x640xf16>, tensor<10x64x640xf16>)
      outs(%13, %13, %13 : tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>)
      attrs = {lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 1, 32, 32, 0]}>} {
  ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f32, %out_3: f32, %out_4: f32):
    %16 = arith.extf %in : f16 to f32
    %17 = arith.extf %in_0 : f16 to f32
    %18 = arith.mulf %16, %17 : f32
    %19 = arith.addf %out, %18 : f32
    %20 = arith.extf %in_1 : f16 to f32
    %21 = arith.mulf %16, %20 : f32
    %22 = arith.addf %out_3, %21 : f32
    %23 = arith.extf %in_2 : f16 to f32
    %24 = arith.mulf %16, %23 : f32
    %25 = arith.addf %out_4, %24 : f32
    linalg.yield %19, %22, %25 : f32, f32, f32
  } -> (tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>)
  %15:3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%14#0, %14#1, %14#2 : tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>)
      outs(%11, %11, %11 : tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f16, %out_2: f16, %out_3: f16):
    %16 = arith.truncf %in : f32 to f16
    %17 = arith.truncf %in_0 : f32 to f16
    %18 = arith.truncf %in_1 : f32 to f16
    linalg.yield %16, %17, %18 : f16, f16, f16
  } -> (tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>)
  return %15#0, %15#1, %15#2 : tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>
}
// CHECK-LABEL: func @horizontal_fusion_consumer_fusion1
//       CHECK:   %[[FORALL:.+]]:3 = scf.forall
//  CHECK-SAME:       shared_outs(%[[OUTS0:[a-zA-Z0-9]+]] =
//  CHECK-SAME:       , %[[OUTS1:[a-zA-Z0-9]+]] =
//  CHECK-SAME:       , %[[OUTS2:[a-zA-Z0-9]+]] =
//       CHECK:     %[[ROOT:.+]]:3 = linalg.generic
//       CHECK:     %[[CONSUMER:.+]]:3 = linalg.generic
//  CHECK-SAME:         ins(%[[ROOT]]#0, %[[ROOT]]#1, %[[ROOT]]#2 :
//   CHECK-DAG:     tensor.parallel_insert_slice %[[CONSUMER]]#0 into %[[OUTS0]]
//   CHECK-DAG:     tensor.parallel_insert_slice %[[CONSUMER]]#1 into %[[OUTS1]]
//   CHECK-DAG:     tensor.parallel_insert_slice %[[CONSUMER]]#2 into %[[OUTS2]]

// -----

// Test 2 of 2 that are testing fusion while considering multiple slices.

func.func @horizontal_fusion_consumer_fusion2(%arg0 : tensor<2x4096x640xi8>,
    %arg1 : tensor<2x640x640xi8>, %arg2 : tensor<2x640x640xi8>) -> tensor<2x4096x640xf16> {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %7 = tensor.empty() : tensor<2x4096x640xf16>
  %8 = tensor.empty() : tensor<2x4096x640xi32>
  %9 = linalg.fill ins(%c0_i32 : i32) outs(%8 : tensor<2x4096x640xi32>) -> tensor<2x4096x640xi32>
  %10:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1, %arg2 : tensor<2x4096x640xi8>, tensor<2x640x640xi8>, tensor<2x640x640xi8>)
      outs(%9, %9 : tensor<2x4096x640xi32>, tensor<2x4096x640xi32>)
      attrs =  {lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 64, 64, 0]}>} {
  ^bb0(%in: i8, %in_0: i8, %in_1: i8, %out: i32, %out_2: i32):
    %12 = arith.extsi %in : i8 to i32
    %13 = arith.extsi %in_0 : i8 to i32
    %14 = arith.muli %12, %13 : i32
    %15 = arith.addi %out, %14 : i32
    %16 = arith.extsi %in_1 : i8 to i32
    %17 = arith.muli %12, %16 : i32
    %18 = arith.addi %out_2, %17 : i32
    linalg.yield %15, %18 : i32, i32
  } -> (tensor<2x4096x640xi32>, tensor<2x4096x640xi32>)
  %11 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%10#1, %10#0 : tensor<2x4096x640xi32>, tensor<2x4096x640xi32>) outs(%7 : tensor<2x4096x640xf16>) {
  ^bb0(%in: i32, %in_0: i32, %out: f16):
    %12 = arith.sitofp %in : i32 to f32
    %13 = arith.truncf %12 : f32 to f16
    %14 = arith.sitofp %in_0 : i32 to f32
    %15 = arith.truncf %14 : f32 to f16
    %16 = arith.addf %13, %15 : f16
    linalg.yield %16 : f16
  } -> tensor<2x4096x640xf16>
  return %11 : tensor<2x4096x640xf16>
}
// CHECK-LABEL: func @horizontal_fusion_consumer_fusion2
//       CHECK:   %[[FORALL:.+]] = scf.forall
//  CHECK-SAME:       shared_outs(%[[OUTS0:[a-zA-Z0-9]+]] =
//       CHECK:     %[[ROOT:.+]]:2 = linalg.generic
//       CHECK:     %[[CONSUMER:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[ROOT]]#1, %[[ROOT]]#0 :
//   CHECK-DAG:     tensor.parallel_insert_slice %[[CONSUMER]] into %[[OUTS0]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
func.func @only_producer_fusion_multiple_result(%arg0: tensor<77x4096xf16>, %arg1: tensor<77x4096xf32>, %arg2: tensor<4096xf16>) -> (tensor<77x4096xf16>, tensor<77x4096xf16>) {
  %cst = arith.constant 9.99999997E-7 : f32
  %cst_0 = arith.constant 4.096000e+03 : f16
  %cst_1 = arith.constant 0.000000e+00 : f16
  %c2_i64 = arith.constant 2 : i64
  %0 = tensor.empty() : tensor<77xf16>
  %1 = tensor.empty() : tensor<77x4096xf16>
  %2 = linalg.fill ins(%cst_1 : f16) outs(%0 : tensor<77xf16>) -> tensor<77xf16>
  %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<77x4096xf16>, tensor<77x4096xf32>) outs(%1 : tensor<77x4096xf16>) {
  ^bb0(%in: f16, %in_2: f32, %out: f16):
    %6 = arith.truncf %in_2 : f32 to f16
    %7 = arith.addf %in, %6 : f16
    linalg.yield %7 : f16
  } -> tensor<77x4096xf16>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%3 : tensor<77x4096xf16>) outs(%2 : tensor<77xf16>)  {
  ^bb0(%in: f16, %out: f16):
    %6 = math.fpowi %in, %c2_i64 : f16, i64
    %7 = arith.addf %6, %out : f16
    linalg.yield %7 : f16
  } -> tensor<77xf16>
  %5 = linalg.generic {indexing_maps = [#map2, #map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg2, %3, %4 : tensor<4096xf16>, tensor<77x4096xf16>, tensor<77xf16>) outs(%1 : tensor<77x4096xf16>) attrs =  {lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 0]}>} {
  ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
    %6 = arith.divf %in_3, %cst_0 : f16
    %7 = arith.truncf %cst : f32 to f16
    %8 = arith.addf %6, %7 : f16
    %9 = math.rsqrt %8 : f16
    %10 = arith.mulf %in_2, %9 : f16
    %11 = arith.mulf %in, %10 : f16
    linalg.yield %11 : f16
  } -> tensor<77x4096xf16>
  return %3, %5 : tensor<77x4096xf16>, tensor<77x4096xf16>
}
// CHECK-LABEL: func @only_producer_fusion_multiple_result
//       CHECK:   %[[RESULT:.+]]:2 = scf.forall
//       CHECK:     linalg.generic
//       CHECK:     linalg.generic
//       CHECK:     linalg.generic
//       CHECK:   return %[[RESULT]]#1, %[[RESULT]]#0

// -----

func.func @multi_slice_fusion_broadcast(%arg0: index, %arg1: tensor<3x?x32xi64>,
     %arg2: tensor<256x32xf32>, %arg3: tensor<32xf32>)
     -> (tensor<3x?x32x32xf32>, tensor<3x?x32x32xf32>) {
  %c32 = arith.constant 32 : index
  %c2_i64 = arith.constant 2 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 3.200000e+01 : f32
  %cst_1 = arith.constant 9.000000e+00 : f32
  %0 = arith.divsi %arg0, %c32 : index
  %1 = affine.apply affine_map<()[s0] -> (s0 floordiv 32)>()[%arg0]
  %2 = tensor.empty(%1) : tensor<3x?x32x32xf32>
  %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg1 : tensor<3x?x32xi64>) outs(%2 : tensor<3x?x32x32xf32>) {
    ^bb0(%in: i64, %out: f32):
      %8 = arith.index_cast %in : i64 to index
      %9 = linalg.index 3 : index
      %extracted = tensor.extract %arg2[%8, %9] : tensor<256x32xf32>
      linalg.yield %extracted : f32
    } -> tensor<3x?x32x32xf32>
  %4 = tensor.empty(%0) : tensor<3x?x32xf32>
  %5 = linalg.fill ins(%cst : f32)outs(%4 : tensor<3x?x32xf32>) -> tensor<3x?x32xf32>
  %6 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%3 : tensor<3x?x32x32xf32>) outs(%5 : tensor<3x?x32xf32>)
      attrs = {lowering_config = #iree_gpu.lowering_config<{reduction = [0, 0, 0, 4], thread = [1, 1, 1, 0], workgroup = [1, 1, 32, 0]}>} {
  ^bb0(%in: f32, %out: f32):
    %8 = math.fpowi %in, %c2_i64 : f32, i64
    %9 = arith.addf %8, %out : f32
    linalg.yield %9 : f32
  } -> tensor<3x?x32xf32>
  %7 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg3, %3, %6 : tensor<32xf32>, tensor<3x?x32x32xf32>, tensor<3x?x32xf32>)
      outs(%2 : tensor<3x?x32x32xf32>) {
  ^bb0(%in: f32, %in_2: f32, %in_3: f32, %out: f32):
    %8 = arith.divf %in_3, %cst_0 : f32
    %9 = arith.addf %8, %cst_1 : f32
    %10 = math.rsqrt %9 : f32
    %11 = arith.mulf %in_2, %10 : f32
    %12 = arith.mulf %in, %11 : f32
    linalg.yield %12 : f32
  } -> tensor<3x?x32x32xf32>
  return %3, %7 : tensor<3x?x32x32xf32>, tensor<3x?x32x32xf32>
}
// CHECK-LABEL: func @multi_slice_fusion_broadcast
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<3x?x32xi64>
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<32xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty
//       CHECK:    %[[RESULT:.+]]:2 = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]])
//  CHECK-SAME:        shared_outs(%[[INIT0:[a-zA-Z0-9]+]] = %[[EMPTY]], %[[INIT1:[a-zA-Z0-9]+]] = %[[EMPTY]])
//   CHECK-DAG:      %[[INIT0_SLICE:.+]] = tensor.extract_slice %[[INIT0]][%[[IV0]], %[[IV1]], 0, 0] [1, 1, 32, 32]
//   CHECK-DAG:      %[[ARG1_SLICE:.+]] = tensor.extract_slice %[[ARG1]][%[[IV0]], %[[IV1]], 0] [1, 1, 32]
//       CHECK:      %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:          ins(%[[ARG1_SLICE]] :
//  CHECK-SAME:          outs(%[[INIT0_SLICE]] :
//       CHECK:      %[[EMPTYTILE:.+]] = tensor.empty() : tensor<1x1x32xf32>
//       CHECK:      %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:          outs(%[[EMPTYTILE]] :
//       CHECK:      %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:          ins(%[[GENERIC0]] :
//  CHECK-SAME:          outs(%[[FILL]] :
//       CHECK:      %[[INIT1_SLICE:.+]] = tensor.extract_slice %[[INIT1]][%[[IV0]], %[[IV1]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1]
//       CHECK:      %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:          ins(%[[ARG3]], %[[GENERIC0]], %[[GENERIC1]] :
//  CHECK-SAME:          outs(%[[INIT1_SLICE]] :
//   CHECK-DAG:      tensor.parallel_insert_slice %[[GENERIC0]] into %[[INIT0]][%[[IV0]], %[[IV1]], 0, 0] [1, 1, 32, 32]
//   CHECK-DAG:      tensor.parallel_insert_slice %[[GENERIC2]] into %[[INIT1]][%[[IV0]], %[[IV1]], 0, 0] [1, 1, 32, 32]
//       CHECK:    return %[[RESULT]]#0, %[[RESULT]]#1

// -----

// Verify that the scf.forall is still generated when only one worker is needed
// for distribution.
func.func @single_trip_forall(%0 : tensor<64x64xf32>, %1 : tensor<64x64xf32>) -> tensor<64x64xf32> {
  %2 = linalg.copy {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>}
      ins(%0 : tensor<64x64xf32>)
      outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %2 : tensor<64x64xf32>
}
//   CHECK-LABEL: func @single_trip_forall
//         CHECK:   scf.forall

// -----

// Verify that the scf.forall is generated and the pack op is outside the op
// without a crash. The consumer fusion is not available because it is not a
// perfect tiling case.

#config = #iree_cpu.lowering_config<distribution = [1, 16]>
#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
func.func @infusible_pack(%arg0 : tensor<30xf32>) -> tensor<5x6xf32> {
  %empty = tensor.empty() : tensor<30xf32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%arg0 : tensor<30xf32>) outs(%empty : tensor<30xf32>)
      attrs = {lowering_config = #config} {
    ^bb0(%b0 : f32, %b1 : f32) :
      %1 = arith.addf %b0, %b0 : f32
      linalg.yield %1 : f32
  } -> tensor<30xf32>
  %empty1 = tensor.empty() : tensor<5x6xf32>
  %pack = linalg.pack %0 outer_dims_perm = [0]
      inner_dims_pos = [0] inner_tiles = [6] into %empty1
      : tensor<30xf32> -> tensor<5x6xf32>
  return %pack : tensor<5x6xf32>
}
// CHECK-LABEL: func @infusible_pack
//       CHECK:   scf.forall
//       CHECK:     linalg.generic
//       CHECK:   scf.forall.in_parallel {
//       CHECK:   linalg.pack

// -----

// Adapted from layer normalization. The graph structure is as follows
//
//              %14
//           /   |   \
//         /    %15   %17
//        |      |   / |
//        |    [%19]   |
//       %21     |    %22
//        |      |     |
//        v      v     v
//
// In particular, %21 and %22 are not users of the "main" tilable
// operation but we still want them to be fused. %19, %21 and %22
// all produce results returned from the function.
//
// Check that everything is fused and that there are three results
// from the loop being produced and returned.
//
// CHECK-LABEL: @multi_result_consumer_fusion
//   CHECK-NOT: linalg.generic
//       CHECK: %[[LOOP:.+]]:3 = scf.forall (%[[I:.+]], %[[J:.+]]) in (16, 256) shared_outs(%[[OUT0:.+]] = %{{.+}}, %[[OUT1:.+]] = %{{.+}}, %[[OUT2:.+]] = %{{.+}})
//       CHECK: %[[v14:.+]] = linalg.generic
//       CHECK:     arith.divf
//       CHECK:   %[[v15:.+]] = linalg.generic
//       CHECK:     arith.subf
//       CHECK:   %[[v17:.+]] = linalg.generic
//       CHECK:     arith.divf
//       CHECK:     math.rsqrt
//       CHECK:   %[[RES0:.+]] = linalg.generic
//       CHECK:     arith.mulf
//       CHECK:     arith.extf
//       CHECK:     arith.mulf
//       CHECK:     arith.extf
//       CHECK:     arith.addf
//       CHECK:     arith.truncf
//       CHECK:   %[[RES1:.+]] = linalg.generic {{.*}} ins(%[[v14]] :
//       CHECK:     arith.truncf
//       CHECK:   %[[RES2:.+]] = linalg.generic {{.*}} ins(%[[v17]] :
//       CHECK:     arith.truncf
//       CHECK:   scf.forall.in_parallel
//       CHECK:     tensor.parallel_insert_slice %[[RES0]] into %[[OUT0]]
//       CHECK:     tensor.parallel_insert_slice %[[RES1]] into %[[OUT1]]
//       CHECK:     tensor.parallel_insert_slice %[[RES2]] into %[[OUT2]]
//   CHECK-NOT: linalg.generic
//       CHECK: return %[[LOOP]]#0, %[[LOOP]]#1, %[[LOOP]]#2
func.func @multi_result_consumer_fusion(
  %6: tensor<16x256x2048xbf16>,
  %7: tensor<2048xbf16>,
  %8: tensor<2048xbf16>,
  %10: tensor<16x256x2048xf32>,
  %13: tensor<16x256xf32>
) -> (
  tensor<16x256x2048xbf16>,
  tensor<16x256xbf16>,
  tensor<16x256xbf16>
) {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 2.048000e+03 : f32
  %c0 = arith.constant 0 : index
  %9 = tensor.empty() : tensor<16x256x2048xf32>
  %11 = tensor.empty() : tensor<16x256xf32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<16x256xf32>) outs(%11 : tensor<16x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    %23 = arith.divf %in, %cst_0 : f32
    linalg.yield %23 : f32
  } -> tensor<16x256xf32>
  %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %14 : tensor<16x256x2048xf32>, tensor<16x256xf32>) outs(%9 : tensor<16x256x2048xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %23 = arith.subf %in, %in_1 : f32
    linalg.yield %23 : f32
  } -> tensor<16x256x2048xf32>
  %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<16x256xf32>) outs(%11 : tensor<16x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    %23 = arith.divf %in, %cst_0 : f32
    %24 = math.rsqrt %23 : f32
    linalg.yield %24 : f32
  } -> tensor<16x256xf32>
  %18 = tensor.empty() : tensor<16x256x2048xbf16>
  %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %17, %7, %8 : tensor<16x256x2048xf32>, tensor<16x256xf32>, tensor<2048xbf16>, tensor<2048xbf16>) outs(%18 : tensor<16x256x2048xbf16>) attrs =  {lowering_config = #iree_gpu.lowering_config<{lane_basis = [[1, 1, 64], [0, 1, 2]], reduction = [0, 0, 256], subgroup_basis = [[1, 1, 1], [0, 1, 2]], thread = [0, 0, 4], workgroup = [1, 1, 0]}>} {
  ^bb0(%in: f32, %in_1: f32, %in_2: bf16, %in_3: bf16, %out: bf16):
    %23 = arith.mulf %in, %in_1 : f32
    %24 = arith.extf %in_2 : bf16 to f32
    %25 = arith.mulf %23, %24 : f32
    %26 = arith.extf %in_3 : bf16 to f32
    %27 = arith.addf %25, %26 : f32
    %28 = arith.truncf %27 : f32 to bf16
    linalg.yield %28 : bf16
  } -> tensor<16x256x2048xbf16>
  %20 = tensor.empty() : tensor<16x256xbf16>
  %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<16x256xf32>) outs(%20 : tensor<16x256xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %23 = arith.truncf %in : f32 to bf16
    linalg.yield %23 : bf16
  } -> tensor<16x256xbf16>
  %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17 : tensor<16x256xf32>) outs(%20 : tensor<16x256xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %23 = arith.truncf %in : f32 to bf16
    linalg.yield %23 : bf16
  } -> tensor<16x256xbf16>
  return %19, %21, %22 : tensor<16x256x2048xbf16>, tensor<16x256xbf16>, tensor<16x256xbf16>
}

// -----

// Adapted from a llama405b reduction dispatch. Tests that all fusable users (the generics with arith.scaling_truncf and arith.truncf) are visited and fused.

// CHECK-LABEL: @multi_fusable_users
//   CHECK-NOT: linalg.generic
//       CHECK: %[[LOOP:.+]]:2 = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (%{{.+}}, 65536) step (1, 256) shared_outs(%[[OUT0:.+]] = %{{.+}}, %[[OUT1:.+]] = %{{.+}})
//       CHECK:   linalg.generic
//       CHECK:     arith.extf
//       CHECK:   linalg.generic
//       CHECK:     math.absf
//       CHECK:     arith.maximumf
//       CHECK:   %[[RES0:.+]] = linalg.generic
//       CHECK:     arith.mulf
//       CHECK:     arith.scaling_truncf
//       CHECK:   %[[RES1:.+]] = linalg.generic
//       CHECK:     arith.mulf
//       CHECK:     arith.truncf
//       CHECK:     arith.bitcast
//       CHECK:   scf.forall.in_parallel
//       CHECK:     tensor.parallel_insert_slice %[[RES0]] into %[[OUT0]]
//       CHECK:     tensor.parallel_insert_slice %[[RES1]] into %[[OUT1]]
//   CHECK-NOT: linalg.generic
//       CHECK: return %[[LOOP]]#1, %[[LOOP]]#0
func.func @multi_fusable_users(%arg0: tensor<?x65536x32xf16>, %arg1: index, %arg2: index, %arg3: index) -> (tensor<?x65536xi8>, tensor<?x65536x32xf4E2M1FN>) {
  %cst = arith.constant 2.500000e-01 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = tensor.empty(%arg1) : tensor<?x65536x32xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<?x65536x32xf16>) outs(%0 : tensor<?x65536x32xf32>) {
  ^bb0(%in: f16, %out: f32):
    %9 = arith.extf %in : f16 to f32
    linalg.yield %9 : f32
  } -> tensor<?x65536x32xf32>
  %2 = tensor.empty(%arg2) : tensor<?x65536xf32>
  %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<?x65536xf32>) -> tensor<?x65536xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1 : tensor<?x65536x32xf32>) outs(%3 : tensor<?x65536xf32>) attrs =  {lowering_config = #iree_gpu.lowering_config<{reduction = [0, 0, 4], thread = [1, 4, 0], workgroup = [1, 256, 0]}>} {
  ^bb0(%in: f32, %out: f32):
    %9 = math.absf %in : f32
    %10 = arith.maximumf %9, %out : f32
    linalg.yield %10 : f32
  } -> tensor<?x65536xf32>
  %5 = tensor.empty(%arg1) : tensor<?x65536x32xf4E2M1FN>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %4 : tensor<?x65536x32xf32>, tensor<?x65536xf32>) outs(%5 : tensor<?x65536x32xf4E2M1FN>) {
  ^bb0(%in: f32, %in_1: f32, %out: f4E2M1FN):
    %9 = arith.mulf %in_1, %cst : f32
    %10 = arith.scaling_truncf %in, %9 : f32, f32 to f4E2M1FN
    linalg.yield %10 : f4E2M1FN
  } -> tensor<?x65536x32xf4E2M1FN>
  %7 = tensor.empty(%arg3) : tensor<?x65536xi8>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<?x65536xf32>) outs(%7 : tensor<?x65536xi8>) {
  ^bb0(%in: f32, %out: i8):
    %9 = arith.mulf %in, %cst : f32
    %10 = arith.truncf %9 : f32 to f8E8M0FNU
    %11 = arith.bitcast %10 : f8E8M0FNU to i8
    linalg.yield %11 : i8
  } -> tensor<?x65536xi8>
  return %8, %6 : tensor<?x65536xi8>, tensor<?x65536x32xf4E2M1FN>
}
//

// -----
func.func @matmul_transposed_reordering_static_on(%arg0 : tensor<8192x4096xf16>,%arg1 : tensor<128256x4096xf16>) -> tensor<8192x128256xf32>
attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64,
{gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %5 = tensor.empty() : tensor<8192x128256xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<8192x128256xf32>) -> tensor<8192x128256xf32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
   iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 , %arg1  : tensor<8192x4096xf16>, tensor<128256x4096xf16>) outs(%6 : tensor<8192x128256xf32>)
   attrs =  {iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_large_f16", tensor>, lowering_config = #iree_gpu.lowering_config<{workgroup = [256, 256, 0],
   workgroup_reordering_strategy = #iree_gpu.conditional_transpose<8,38>}>} {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %8 = arith.extf %in : f16 to f32
    %9 = arith.extf %in_0 : f16 to f32
    %10 = arith.mulf %8, %9 : f32
    %11 = arith.addf %out, %10 : f32
    linalg.yield %11 : f32
  } -> tensor<8192x128256xf32>
  return %7 : tensor<8192x128256xf32>
}
// CHECK-LABEL: @matmul_transposed_reordering_static_on
//       CHECK: scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (128256, 8192) step (256, 256) shared_outs(%[[OUT0:.+]] = %{{.+}})
//       CHECK:   tensor.extract_slice %{{.+}}[%[[J]], 0]
//       CHECK:   tensor.extract_slice %{{.+}}[%[[I]], 0]
//       CHECK:   tensor.extract_slice %[[OUT0]][%[[J]], %[[I]]]
//       CHECK:   %[[RES:.+]] = linalg.generic
//       CHECK:   scf.forall.in_parallel
//       CHECK:     tensor.parallel_insert_slice %[[RES]] into %[[OUT0]][%[[J]], %[[I]]]
//   CHECK: {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

func.func @matmul_transposed_reordering_static_no_reordering(%arg0 : tensor<8192x4096xf16>,%arg1 : tensor<128256x4096xf16>) -> tensor<8192x128256xf32>
attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64,
{gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %5 = tensor.empty() : tensor<8192x128256xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<8192x128256xf32>) -> tensor<8192x128256xf32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
   iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 , %arg1  : tensor<8192x4096xf16>, tensor<128256x4096xf16>) outs(%6 : tensor<8192x128256xf32>)
   attrs =  {iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_large_f16", tensor>, lowering_config = #iree_gpu.lowering_config<{workgroup = [256, 256, 0]}>} {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %8 = arith.extf %in : f16 to f32
    %9 = arith.extf %in_0 : f16 to f32
    %10 = arith.mulf %8, %9 : f32
    %11 = arith.addf %out, %10 : f32
    linalg.yield %11 : f32
  } -> tensor<8192x128256xf32>
  return %7 : tensor<8192x128256xf32>
}
// CHECK-LABEL: @matmul_transposed_reordering_static_no_reordering
//       CHECK: scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (8192, 128256) step (256, 256) shared_outs(%[[OUT0:.+]] = %{{.+}})
//       CHECK:   tensor.extract_slice %{{.+}}[%[[I]], 0]
//       CHECK:   tensor.extract_slice %{{.+}}[%[[J]], 0]
//       CHECK:   tensor.extract_slice %[[OUT0]][%[[I]], %[[J]]]
//       CHECK:   %[[RES:.+]] = linalg.generic
//       CHECK:   scf.forall.in_parallel
//       CHECK:     tensor.parallel_insert_slice %[[RES]] into %[[OUT0]][%[[I]], %[[J]]]
//   CHECK: {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

func.func @matmul_transposed_reordering_static_off(%arg0 : tensor<128256x4096xf16>,%arg1 : tensor<8192x4096xf16>) -> tensor<128256x8192xf32>
attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64,
{gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %7 = tensor.empty() : tensor<128256x8192xf32>
  %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<128256x8192xf32>) -> tensor<128256x8192xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]}
  ins(%arg0, %arg1 : tensor<128256x4096xf16>, tensor<8192x4096xf16>) outs(%8 : tensor<128256x8192xf32>) attrs =  {iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_large_f16", tensor>,
  lowering_config = #iree_gpu.lowering_config<{workgroup = [256, 256, 0], workgroup_reordering_strategy = #iree_gpu.conditional_transpose<8, 38>}>} {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %10 = arith.extf %in : f16 to f32
    %11 = arith.extf %in_0 : f16 to f32
    %12 = arith.mulf %10, %11 : f32
    %13 = arith.addf %out, %12 : f32
    linalg.yield %13 : f32
  } -> tensor<128256x8192xf32>
  return %9 : tensor<128256x8192xf32>
}
// CHECK-LABEL: @matmul_transposed_reordering_static_off
//       CHECK: scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (128256, 8192) step (256, 256) shared_outs(%[[OUT0:.+]] = %{{.+}})
//       CHECK:   tensor.extract_slice %{{.+}}[%[[I]], 0]
//       CHECK:   tensor.extract_slice %{{.+}}[%[[J]], 0]
//       CHECK:   tensor.extract_slice %[[OUT0]][%[[I]], %[[J]]]
//       CHECK:   %[[RES:.+]] = linalg.generic
//       CHECK:   scf.forall.in_parallel
//       CHECK:     tensor.parallel_insert_slice %[[RES]] into %[[OUT0]][%[[I]], %[[J]]]
//   CHECK: {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}


func.func @matmul_transposed_reordering_dynamic(%arg0 : tensor<?x256x4096xf16>,%arg1 : tensor<8192x4096xf16>) -> tensor<?x256x8192xf32>
attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64,
{gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>} {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x256x4096xf16>
  %23 = tensor.empty(%dim0) : tensor<?x256x8192xf32>
  %24 = linalg.fill ins(%cst : f32) outs(%23 : tensor<?x256x8192xf32>) -> tensor<?x256x8192xf32>
  %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
  iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x256x4096xf16>, tensor<8192x4096xf16>) outs(%24 : tensor<?x256x8192xf32>)
  attrs =  {iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_large_f16_expanded", tensor>,
  lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 256, 256, 0], workgroup_reordering_strategy = #iree_gpu.conditional_transpose<8, 38>}>} {
  ^bb0(%in: f16, %in_1: f16, %out: f32):
    %28 = arith.extf %in : f16 to f32
    %29 = arith.extf %in_1 : f16 to f32
    %30 = arith.mulf %28, %29 : f32
    %31 = arith.addf %out, %30 : f32
    linalg.yield %31 : f32
  } -> tensor<?x256x8192xf32>
  return %27 : tensor<?x256x8192xf32>
}
// CHECK-LABEL: @matmul_transposed_reordering_dynamic
//       CHECK:   %[[COND:.+]] = arith.cmpi ult, %{{.+}}, %c33 : index
//       CHECK:   %[[BOUND0:.+]] = arith.select %[[COND]], %c8192, %{{.+}} : index
//       CHECK:   %[[BOUND1:.+]] = arith.select %[[COND]], %{{.+}}, %c8192 : index
//       CHECK:   %[[STEP0:.+]] = arith.select %[[COND]], %c256, %c1 : index
//       CHECK:   %[[STEP1:.+]] = arith.select %[[COND]], %c1, %c256 : index
//       CHECK: scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (%[[BOUND0]], %[[BOUND1]]) step (%[[STEP0]], %[[STEP1]]) shared_outs(%[[OUT0:.+]] = %{{.+}})
//       CHECK:   %[[OFFSET0:.+]] = arith.select %[[COND]], %[[J]], %[[I]] : index
//       CHECK:   %[[OFFSET1:.+]] = arith.select %[[COND]], %[[I]], %[[J]] : index
//       CHECK:   tensor.extract_slice %{{.+}}[%[[OFFSET0]], 0, 0]
//       CHECK:   tensor.extract_slice %{{.+}}[%[[OFFSET1]], 0]
//       CHECK:   %[[RES:.+]] = linalg.generic
//       CHECK:   scf.forall.in_parallel
//       CHECK:     tensor.parallel_insert_slice %[[RES]] into %[[OUT0]][%[[OFFSET0]], 0, %[[OFFSET1]]]
//   CHECK: {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
