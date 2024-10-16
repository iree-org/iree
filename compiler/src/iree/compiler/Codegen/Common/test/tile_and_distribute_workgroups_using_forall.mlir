// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups-using-forall-op, cse))" --mlir-print-local-scope --split-input-file %s | FileCheck %s

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
  memref.assume_alignment %arg0, 64 : memref<?x?x?xf32>
  memref.assume_alignment %arg1, 64 : memref<?x?x?xf32>
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
    %arg0 : tensor<?x16x32x128xf16>, %arg1 : tensor<4096x32x28xf16>,
    %arg2 : tensor<?x16x4096xf16>) -> tensor<?x16x4096xf16> {
   %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
      ins(%arg0, %arg1 : tensor<?x16x32x128xf16>, tensor<4096x32x28xf16>)
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
        #iree_codegen.lowering_config<tile_sizes = [[16, 64, 0]]>} {
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
