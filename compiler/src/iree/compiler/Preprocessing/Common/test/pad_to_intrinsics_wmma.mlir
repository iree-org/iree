// RUN: iree-opt --split-input-file %s --iree-gpu-test-target=gfx1100 --pass-pipeline="builtin.module(iree-preprocessing-pad-to-intrinsics,func.func(canonicalize))" | FileCheck %s
// RUN: iree-opt --split-input-file %s --iree-gpu-test-target=gfx1100 --pass-pipeline="builtin.module(iree-preprocessing-pad-to-intrinsics{pad-target-type=conv},func.func(canonicalize))" | FileCheck %s -check-prefix=CONVOLUTION
// RUN: iree-opt --split-input-file %s --iree-gpu-test-target=gfx1100 --pass-pipeline="builtin.module(iree-preprocessing-pad-to-intrinsics{pad-target-type=contraction},func.func(canonicalize))" | FileCheck %s -check-prefix=CONTRACT

//       CHECK: func.func @matmul_static(
//  CHECK-SAME:    %[[ARG0:.+]]: tensor<10x20xf16>,
//  CHECK-SAME:    %[[ARG1:.+]]: tensor<20x30xf16>,
//  CHECK-SAME:    %[[ARG2:.+]]: tensor<10x30xf16>)
func.func @matmul_static(%arg0 : tensor<10x20xf16>, %arg1 : tensor<20x30xf16>, %arg2 : tensor<10x30xf16>) -> tensor<10x30xf16> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<10x20xf16>, tensor<20x30xf16>)
        outs(%arg2 : tensor<10x30xf16>) -> tensor<10x30xf16>
    return %0 : tensor<10x30xf16>
}

// CHECK:      %[[CST0:.+]] = arith.constant 0.0{{.*}} : f16
// CHECK:      %[[PAD_LHS:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[6, 12]
// CHECK:      %[[PAD_RHS:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[12, 2]
// CHECK:      %[[PAD_INIT:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[6, 2]
// CHECK:      %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:                  ins(%[[PAD_LHS]], %[[PAD_RHS]] : tensor<16x32xf16>, tensor<32x32xf16>
// CHECK-SAME:                  outs(%[[PAD_INIT]] : tensor<16x32xf16>
// CHECK:      %[[EXTRACT:.+]] = tensor.extract_slice %[[MATMUL]][0, 0] [10, 30] [1, 1] : tensor<16x32xf16> to tensor<10x30xf16>
// CHECK:      return %[[EXTRACT]]

// CONVOLUTION-NOT: tensor.pad {{.*}} low[0, 0] high[6, 12]
// CONVOLUTION-NOT: tensor.pad {{.*}} low[0, 0] high[12, 2]
// CONVOLUTION-NOT: tensor.pad {{.*}} low[0, 0] high[6, 2]

// CONTRACT:        tensor.pad {{.*}} low[0, 0] high[6, 12]
// CONTRACT:        tensor.pad {{.*}} low[0, 0] high[12, 2]
// CONTRACT:        tensor.pad {{.*}} low[0, 0] high[6, 2]

// -----

// Good test to ensure reassoc, new dims, and iterator types works on permuted operations.

//   CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (-s0 + (s0 ceildiv 16) * 16)>
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
//       CHECK: func.func @mmtb_dynamic_k_n(
//  CHECK-SAME:    %[[ARG0:.+]]: tensor<10x?xf16>,
//  CHECK-SAME:    %[[ARG1:.+]]: tensor<?x?xf16>,
//  CHECK-SAME:    %[[ARG2:.+]]: tensor<10x?xf16>)
func.func @mmtb_dynamic_k_n(%arg0 : tensor<10x?xf16>, %arg1 : tensor<?x?xf16>, %arg2 : tensor<10x?xf16>) -> tensor<10x?xf16> {
    %0 = linalg.matmul
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
        ]
        ins(%arg0, %arg1 : tensor<10x?xf16>, tensor<?x?xf16>)
        outs(%arg2 : tensor<10x?xf16>) -> tensor<10x?xf16>
    return %0 : tensor<10x?xf16>
}

// CHECK:      %[[CST0:.+]] = arith.constant 0.0{{.*}} : f16
// CHECK:      %[[DIM_N:.+]] = tensor.dim %[[ARG1]], %c0 : tensor<?x?xf16>
// CHECK:      %[[PADSIZE_N:.+]] = affine.apply #[[MAP]]()[%[[DIM_N]]]
// CHECK:      %[[DIM_K:.+]] = tensor.dim %[[ARG0]], %c1 : tensor<10x?xf16>
// CHECK:      %[[PADSIZE_K:.+]] = affine.apply #[[MAP]]()[%[[DIM_K]]]
// CHECK:      %[[PAD_LHS:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[6, %[[PADSIZE_K]]]
// CHECK:      %[[PAD_RHS:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[PADSIZE_N]], %[[PADSIZE_K]]]
// CHECK:      %[[PAD_INIT:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[6, %[[PADSIZE_N]]]
// CHECK:      %[[EXP_LHS:.+]] = tensor.expand_shape %[[PAD_LHS]] {{\[}}[0], [1, 2]] output_shape {{.*}} : tensor<16x?xf16> into tensor<16x?x16xf16>
// CHECK:      %[[EXP_RHS:.+]] = tensor.expand_shape %[[PAD_RHS]] {{\[}}[0, 1], [2, 3]] output_shape {{.*}} : tensor<?x?xf16> into tensor<?x16x?x16xf16>
// CHECK:      %[[EXP_INIT:.+]] = tensor.expand_shape %[[PAD_INIT]] {{\[}}[0], [1, 2]] output_shape {{.*}} : tensor<16x?xf16> into tensor<16x?x16xf16>
// CHECK:      %[[MMTB:.+]] = linalg.generic
// CHECK-SAME:                    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:                    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME:                    ins(%[[EXP_LHS]], %[[EXP_RHS]]
// CHECK-SAME:                    outs(%[[EXP_INIT]]
// CHECK:      %[[COLLAPSE:.+]] = tensor.collapse_shape %[[MMTB]] {{\[}}[0], [1, 2]] : tensor<16x?x16xf16> into tensor<16x?xf16>
// CHECK:      %[[DIM_N1:.+]] = tensor.dim %[[ARG2]], %c1 : tensor<10x?xf16>
// CHECK:      %[[EXTRACT:.+]] = tensor.extract_slice %[[COLLAPSE]][0, 0] [10, %[[DIM_N1]]] [1, 1] : tensor<16x?xf16> to tensor<10x?xf16>
// CHECK:      return %[[EXTRACT]]

// -----

//   CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (-s0 + (s0 ceildiv 16) * 16)>
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (-s0 + s1 + (s0 ceildiv 16) * 16)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d5)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d3)>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
//       CHECK: func.func @bmm_dynamic_m_k(
//  CHECK-SAME:    %[[ARG0:.+]]: tensor<32x?x?xf16>,
//  CHECK-SAME:    %[[ARG1:.+]]: tensor<32x?x128xf16>)
func.func @bmm_dynamic_m_k(%arg0: tensor<32x?x?xf16>, %arg1: tensor<32x?x128xf16>) -> tensor<32x?x128xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c1 : tensor<32x?x?xf16>
  %0 = tensor.empty(%dim) : tensor<32x?x128xf16>
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<32x?x128xf16>) -> tensor<32x?x128xf16>
  %2 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<32x?x?xf16>, tensor<32x?x128xf16>) outs(%1 : tensor<32x?x128xf16>) -> tensor<32x?x128xf16>
  return %2 : tensor<32x?x128xf16>
}

// CHECK:      %[[CST0:.+]] = arith.constant 0.0{{.*}} : f16
// CHECK:      %[[DIM_M0:.+]] = tensor.dim %[[ARG0]], %c1 : tensor<32x?x?xf16>
// CHECK:      %[[DIM_M1:.+]] = tensor.dim %[[ARG0]], %c1 : tensor<32x?x?xf16>
// CHECK:      %[[PADSIZE_M:.+]] = affine.apply #[[MAP]]()[%[[DIM_M1]]]
// CHECK:      %[[DIM_K:.+]] = tensor.dim %[[ARG0]], %c2 : tensor<32x?x?xf16>
// CHECK:      %[[PADSIZE_K:.+]] = affine.apply #[[MAP]]()[%[[DIM_K]]]
// CHECK:      %[[PAD_LHS:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0] high[0, %[[PADSIZE_M]], %[[PADSIZE_K]]]
// CHECK:      %[[PAD_RHS:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0] high[0, %[[PADSIZE_K]], 0]
// CHECK:      %[[PADDED_M:.+]] = affine.apply #[[MAP0]]()[%[[DIM_M1]], %[[DIM_M0]]]
// CHECK:      %[[INIT:.+]] = tensor.empty(%[[PADDED_M]]) : tensor<32x?x128xf16>
// CHECK:      %[[EXP_LHS:.+]] = tensor.expand_shape %[[PAD_LHS]] {{\[}}[0], [1, 2], [3, 4]] output_shape {{.*}} : tensor<32x?x?xf16> into tensor<32x?x16x?x16xf16>
// CHECK:      %[[EXP_RHS:.+]] = tensor.expand_shape %[[PAD_RHS]] {{\[}}[0], [1, 2], [3]] output_shape {{.*}} : tensor<32x?x128xf16> into tensor<32x?x16x128xf16>
// CHECK:      %[[EXP_INIT:.+]] = tensor.expand_shape %[[INIT]] {{\[}}[0], [1, 2], [3]] output_shape {{.*}} : tensor<32x?x128xf16> into tensor<32x?x16x128xf16>
// CHECK:      %[[FILL:.+]] = linalg.fill {{.*}} outs(%[[EXP_INIT]] : tensor<32x?x16x128xf16>)
// CHECK:      %[[EXP_BMM:.+]] = linalg.generic
// CHECK-SAME:                    indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]
// CHECK-SAME:                    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME:                    ins(%[[EXP_LHS]], %[[EXP_RHS]]
// CHECK-SAME:                    outs(%[[FILL]]
// CHECK:      %[[COLLAPSE:.+]] = tensor.collapse_shape %[[EXP_BMM]] {{\[}}[0], [1, 2], [3]]
// CHECK:      %[[EXTRACT:.+]] = tensor.extract_slice %[[COLLAPSE]][0, 0, 0] [32, %[[DIM_M0]], 128] [1, 1, 1]
// CHECK:      return %[[EXTRACT]]

// -----

//   CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (-s0 + (s0 ceildiv 16) * 16)>
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (-s0 + s1 + (s0 ceildiv 16) * 16)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
//       CHECK: func.func @dequant_gemm_dynamic_m(
//  CHECK-SAME:    %[[ARG0:.+]]: tensor<4096x32x128xi4>,
//  CHECK-SAME:    %[[ARG1:.+]]: tensor<4096x32xf16>, %[[ARG2:.+]]: tensor<4096x32xf16>,
//  CHECK-SAME:    %[[ARG3:.+]]: tensor<?x32x128xf16>)
func.func @dequant_gemm_dynamic_m(%arg0: tensor<4096x32x128xi4>, %arg1: tensor<4096x32xf16>, %arg2: tensor<4096x32xf16>, %arg3: tensor<?x32x128xf16>) -> tensor<?x4096xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %dim = tensor.dim %arg3, %c0 : tensor<?x32x128xf16>
  %1 = tensor.empty(%dim) : tensor<?x4096xf16>
  %2 = tensor.empty() : tensor<4096x32x128xf16>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<4096x32x128xi4>, tensor<4096x32xf16>, tensor<4096x32xf16>) outs(%2 : tensor<4096x32x128xf16>) {
  ^bb0(%in: i4, %in_1: f16, %in_2: f16, %out: f16):
    %6 = arith.extui %in : i4 to i32
    %7 = arith.uitofp %6 : i32 to f16
    %8 = arith.subf %7, %in_2 : f16
    %9 = arith.mulf %8, %in_1 : f16
    linalg.yield %9 : f16
  } -> tensor<4096x32x128xf16>
  %4 = linalg.fill ins(%cst : f16) outs(%1 : tensor<?x4096xf16>) -> tensor<?x4096xf16>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%arg3, %3 : tensor<?x32x128xf16>, tensor<4096x32x128xf16>) outs(%4 : tensor<?x4096xf16>) {
  ^bb0(%in: f16, %in_1: f16, %out: f16):
    %6 = arith.mulf %in, %in_1 : f16
    %7 = arith.addf %6, %out : f16
    linalg.yield %7 : f16
  } -> tensor<?x4096xf16>
  return %5 : tensor<?x4096xf16>
}

// CHECK:      %[[CST0:.+]] = arith.constant 0.0{{.*}} : f16
// CHECK:      %[[DIM_M0:.+]] = tensor.dim %[[ARG3]], %c0 : tensor<?x32x128xf16>
// CHECK:      %[[DEQUANT:.+]] = linalg.generic
// CHECK-SAME:                    iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:                    ins(%[[ARG0]], %[[ARG1]], %[[ARG2]]
// CHECK:      %[[DIM_M1:.+]] = tensor.dim %[[ARG3]], %c0 : tensor<?x32x128xf16>
// CHECK:      %[[PADSIZE_M:.+]] = affine.apply #[[MAP]]()[%[[DIM_M1]]]
// CHECK:      %[[PAD_LHS:.+]] = tensor.pad %[[ARG3]] low[0, 0, 0] high[%[[PADSIZE_M]], 0, 0]
// CHECK:      %[[EXP_LHS:.+]] = tensor.expand_shape %[[PAD_LHS]] {{\[}}[0, 1], [2], [3]] output_shape {{.*}} : tensor<?x32x128xf16> into tensor<?x16x32x128xf16>
// CHECK:      %[[FILL:.+]] = linalg.fill {{.*}}-> tensor<?x16x4096xf16>
// CHECK:      %[[EXP_GEMM:.+]] = linalg.generic
// CHECK-SAME:                    indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]
// CHECK-SAME:                    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME:                    ins(%[[EXP_LHS]], %[[DEQUANT]]
// CHECK-SAME:                    outs(%[[FILL]]
// CHECK:      %[[COLLAPSE:.+]] = tensor.collapse_shape %[[EXP_GEMM]] {{\[}}[0, 1], [2]]
// CHECK:      %[[EXTRACT:.+]] = tensor.extract_slice %[[COLLAPSE]][0, 0] [%[[DIM_M0]], 4096] [1, 1]
// CHECK:      return %[[EXTRACT]]

// CONVOLUTION-NOT: tensor.pad {{.*}} low[0, 0, 0]
// CONVOLUTION-NOT: tensor.expand_shape {{.*}} {{\[}}[0, 1], [2], [3]] output_shape {{.*}} : tensor<?x32x128xf16> into tensor<?x16x32x128xf16>

// CONTRACT:        tensor.pad {{.*}} low[0, 0, 0]
// CONTRACT:        tensor.expand_shape {{.*}} {{\[}}[0, 1], [2], [3]] output_shape {{.*}} : tensor<?x32x128xf16> into tensor<?x16x32x128xf16>
