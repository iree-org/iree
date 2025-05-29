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

func.func @gemm_unit_M_unit_N(%arg0 : tensor<1x1xf32>, %arg1 : tensor<1x1xf32>,
    %arg2 : tensor<1x1xf32>) -> tensor<1x1xf32> {
  %0 = linalg.matmul {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 64]]>}
      ins(%arg0, %arg1 : tensor<1x1xf32>, tensor<1x1xf32>)
      outs(%arg2 : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @gemm_unit_M_unit_N(
//   CHECK-NOT:   scf.forall

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

// This just verifies that constant dim propagation works as expected after tiling.
func.func @generate_no_distribution(%arg0 : tensor<16xf16>) -> tensor<16xf16> {
   %empty = tensor.empty() : tensor<16xf16>
   %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%arg0 : tensor<16xf16>) outs(%empty : tensor<16xf16>)
      attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[16]]>} {
    ^bb0(%b0: f16, %b1: f16):
      %1 = arith.mulf %b0, %b0 : f16
      linalg.yield %1 : f16
  } -> tensor<16xf16>
  return %0 : tensor<16xf16>
}
// CHECK-LABEL: func @generate_no_distribution(
//   CHECK-NOT:   scf.forall

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
  %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]
  } ins(%0 : tensor<12xf32>) outs(%2 : tensor<12xf32>) attrs =  {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[4]]>} {
  ^bb0(%in: f32, %out: f32):
    %6 = math.sqrt %in : f32
    linalg.yield %6 : f32
  } -> tensor<12xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]
  } ins(%1 : tensor<12xf32>) outs(%2 : tensor<12xf32>) {
  ^bb0(%in: f32, %out: f32):
    %7 = math.sqrt %in : f32
    linalg.yield %7 : f32
  } -> tensor<12xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]
  } ins(%3, %4 : tensor<12xf32>, tensor<12xf32>) outs(%2 : tensor<12xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %8 = arith.addf %in, %in_0 : f32
    linalg.yield %8 : f32
  } -> tensor<12xf32>
  return %5 : tensor<12xf32>
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

// Test 1 of 2 that are testing a work-around for SSA violation issue with consumer fusion upstream.

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

// Test 2 of 2 that are testing a work-around for SSA violation issue with consumer fusion upstream.

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
