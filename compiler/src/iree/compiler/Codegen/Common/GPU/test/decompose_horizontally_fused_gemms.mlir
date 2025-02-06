// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-decompose-horizontally-fused-gemms))" --split-input-file --mlir-print-local-scope %s | FileCheck %s

func.func @fused_contraction_1(%arg0: tensor<2x4096x640xf16>,
    %arg1: tensor<10x64x640xf16>, %arg2: tensor<10x64x640xf16>,
    %arg3: tensor<10x64x640xf16>)
    -> (tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>) {
    %0 = tensor.empty() : tensor<2x10x4096x64xf16>
    %1 = tensor.empty() : tensor<2x10x4096x64xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<2x10x4096x64xf32>) -> tensor<2x10x4096x64xf32>
    %3:3 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%arg0, %arg1, %arg2, %arg3 : tensor<2x4096x640xf16>, tensor<10x64x640xf16>, tensor<10x64x640xf16>, tensor<10x64x640xf16>)
        outs(%2, %2, %2 : tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>)
        attrs = {
            lowering_config = #iree_gpu.lowering_config<{
                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1, 2, 3],
                reduction = [0, 0, 0, 0, 128], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64,
                workgroup = [1, 1, 32, 32, 0]}>} {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f32, %out_3: f32, %out_4: f32):
      %7 = arith.extf %in : f16 to f32
      %8 = arith.extf %in_0 : f16 to f32
      %9 = arith.mulf %7, %8 : f32
      %10 = arith.addf %out, %9 : f32
      %11 = arith.extf %in_1 : f16 to f32
      %12 = arith.mulf %7, %11 : f32
      %13 = arith.addf %out_3, %12 : f32
      %14 = arith.extf %in_2 : f16 to f32
      %15 = arith.mulf %7, %14 : f32
      %16 = arith.addf %out_4, %15 : f32
      linalg.yield %10, %13, %16 : f32, f32, f32
  } -> (tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>)
  return %3#0, %3#1, %3#2 : tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>
}
// CHECK-LABEL: func @fused_contraction_1
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x4096x640xf16>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<10x64x640xf16>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<10x64x640xf16>
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<10x64x640xf16>
//       CHECK:   %[[FILL:.+]] = linalg.fill
//       CHECK:   %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
//  CHECK-SAME:                        affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
//  CHECK-SAME:                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//  CHECK-SAME:       iree_gpu.lowering_config
//  CHECK-SAME:           mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16
//  CHECK-SAME:           promote_operands = [0, 1]
//  CHECK-SAME:           reduction = [0, 0, 0, 0, 128]
//  CHECK-SAME:           subgroup_m_count = 2
//  CHECK-SAME:           subgroup_n_count = 2
//  CHECK-SAME:           workgroup = [1, 1, 32, 32, 0]
//       CHECK:     ^bb0(%[[B0_0:[a-zA-Z0-9_]+]]: f16, %[[B1_0:[a-zA-Z0-9_]+]]: f16, %[[B2_0:[a-zA-Z0-9_]+]]: f32
//   CHECK-DAG:       %[[LHS_0:.+]] = arith.extf %[[B0_0]]
//   CHECK-DAG:       %[[RHS_0:.+]] = arith.extf %[[B1_0]]
//   CHECK-DAG:       %[[MUL_0:.+]] = arith.mulf %[[LHS_0]], %[[RHS_0]]
//   CHECK-DAG:       %[[ADD_0:.+]] = arith.addf %[[B2_0]], %[[MUL_0]]
//   CHECK-DAG:       linalg.yield %[[ADD_0]]
//       CHECK:   %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
//  CHECK-SAME:                        affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
//  CHECK-SAME:                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//  CHECK-SAME:       iree_gpu.lowering_config
//  CHECK-SAME:           mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16
//  CHECK-SAME:           promote_operands = [0, 1]
//  CHECK-SAME:           reduction = [0, 0, 0, 0, 128]
//  CHECK-SAME:           subgroup_m_count = 2
//  CHECK-SAME:           subgroup_n_count = 2
//  CHECK-SAME:           workgroup = [1, 1, 32, 32, 0]
//       CHECK:     ^bb0(%[[B0_1:[a-zA-Z0-9_]+]]: f16, %[[B1_1:[a-zA-Z0-9_]+]]: f16, %[[B2_1:[a-zA-Z0-9_]+]]: f32
//   CHECK-DAG:       %[[LHS_1:.+]] = arith.extf %[[B0_1]]
//   CHECK-DAG:       %[[RHS_1:.+]] = arith.extf %[[B1_1]]
//   CHECK-DAG:       %[[MUL_1:.+]] = arith.mulf %[[LHS_1]], %[[RHS_1]]
//   CHECK-DAG:       %[[ADD_1:.+]] = arith.addf %[[B2_1]], %[[MUL_1]]
//   CHECK-DAG:       linalg.yield %[[ADD_1]]
//       CHECK:   %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
//  CHECK-SAME:                        affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
//  CHECK-SAME:                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG3]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//  CHECK-SAME:       iree_gpu.lowering_config
//  CHECK-SAME:           mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16
//  CHECK-SAME:           promote_operands = [0, 1]
//  CHECK-SAME:           reduction = [0, 0, 0, 0, 128]
//  CHECK-SAME:           subgroup_m_count = 2
//  CHECK-SAME:           subgroup_n_count = 2
//  CHECK-SAME:           workgroup = [1, 1, 32, 32, 0]
//       CHECK:     ^bb0(%[[B0_2:[a-zA-Z0-9_]+]]: f16, %[[B1_2:[a-zA-Z0-9_]+]]: f16, %[[B2_2:[a-zA-Z0-9_]+]]: f32
//   CHECK-DAG:       %[[LHS_2:.+]] = arith.extf %[[B0_2]]
//   CHECK-DAG:       %[[RHS_2:.+]] = arith.extf %[[B1_2]]
//   CHECK-DAG:       %[[MUL_2:.+]] = arith.mulf %[[LHS_2]], %[[RHS_2]]
//   CHECK-DAG:       %[[ADD_2:.+]] = arith.addf %[[B2_2]], %[[MUL_2]]
//   CHECK-DAG:       linalg.yield %[[ADD_2]]
//       CHECK:   return %[[GENERIC0]], %[[GENERIC1]], %[[GENERIC2]]

// -----

func.func @fused_contraction_2(%arg0: tensor<4096x640xf32>,
    %arg1: tensor<640x640xf32>, %arg2: tensor<640x640xf32>, %arg3: tensor<640x640xf32>)
    -> (tensor<4096x640xf32>, tensor<4096x640xf32>, tensor<4096x640xf32>) {
    %0 = tensor.empty() : tensor<4096x640xf32>
    %1 = tensor.empty() : tensor<4096x640xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<4096x640xf32>) -> tensor<4096x640xf32>
    %3:3 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                         affine_map<(d0, d1, d2) -> (d2, d1)>,
                         affine_map<(d0, d1, d2) -> (d2, d1)>,
                         affine_map<(d0, d1, d2) -> (d2, d1)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%arg0, %arg1, %arg2, %arg3 : tensor<4096x640xf32>, tensor<640x640xf32>, tensor<640x640xf32>, tensor<640x640xf32>)
        outs(%2, %2, %2 : tensor<4096x640xf32>, tensor<4096x640xf32>, tensor<4096x640xf32>)
        attrs = {
            lowering_config = #iree_gpu.lowering_config<{
                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>, promote_operands = [0, 1, 2, 3],
                reduction = [0, 0, 16], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [32, 64, 0]}>} {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32, %out_3: f32, %out_4: f32):
      %4 = arith.mulf %in, %in_0 : f32
      %5 = arith.addf %out, %4 : f32
      %6 = arith.mulf %in, %in_1 : f32
      %7 = arith.addf %out_3, %6 : f32
      %8 = arith.mulf %in, %in_2 : f32
      %9 = arith.addf %out_4, %8 : f32
      linalg.yield %5, %7, %9 : f32, f32, f32
  } -> (tensor<4096x640xf32>, tensor<4096x640xf32>, tensor<4096x640xf32>)
  return %3#0, %3#1, %3#2 : tensor<4096x640xf32>, tensor<4096x640xf32>, tensor<4096x640xf32>
}
// CHECK-LABEL: func @fused_contraction_2
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<4096x640xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<640x640xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<640x640xf32>
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<640x640xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill
//       CHECK:   %[[GENERIC0:.+]] = linalg.generic
//       CHECK:     ^bb0(%[[B0_0:[a-zA-Z0-9_]+]]: f32, %[[B1_0:[a-zA-Z0-9_]+]]: f32, %[[B2_0:[a-zA-Z0-9_]+]]: f32
//   CHECK-DAG:       %[[MUL_0:.+]] = arith.mulf %[[B0_0]], %[[B1_0]]
//   CHECK-DAG:       %[[ADD_0:.+]] = arith.addf %[[B2_0]], %[[MUL_0]]
//   CHECK-DAG:       linalg.yield %[[ADD_0]]
//       CHECK:   %[[GENERIC1:.+]] = linalg.generic
//       CHECK:     ^bb0(%[[B0_1:[a-zA-Z0-9_]+]]: f32, %[[B1_1:[a-zA-Z0-9_]+]]: f32, %[[B2_1:[a-zA-Z0-9_]+]]: f32
//   CHECK-DAG:       %[[MUL_1:.+]] = arith.mulf %[[B0_1]], %[[B1_1]]
//   CHECK-DAG:       %[[ADD_1:.+]] = arith.addf %[[B2_1]], %[[MUL_1]]
//   CHECK-DAG:       linalg.yield %[[ADD_1]]
//       CHECK:   %[[GENERIC2:.+]] = linalg.generic
//       CHECK:     ^bb0(%[[B0_2:[a-zA-Z0-9_]+]]: f32, %[[B1_2:[a-zA-Z0-9_]+]]: f32, %[[B2_2:[a-zA-Z0-9_]+]]: f32
//   CHECK-DAG:       %[[MUL_2:.+]] = arith.mulf %[[B0_2]], %[[B1_2]]
//   CHECK-DAG:       %[[ADD_2:.+]] = arith.addf %[[B2_2]], %[[MUL_2]]
//   CHECK-DAG:       linalg.yield %[[ADD_2]]
//       CHECK:   return %[[GENERIC0]], %[[GENERIC1]], %[[GENERIC2]]
