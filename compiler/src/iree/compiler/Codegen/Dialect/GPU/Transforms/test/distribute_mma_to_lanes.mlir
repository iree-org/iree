// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-gpu-distribute-mma-to-lanes, canonicalize, cse))' --split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
module {
  func.func @matmul_16x16x16(%arg0: tensor<8x2x16x16xf16>, %arg1: tensor<8x2x16x16xf16>, %arg2: tensor<2x2x16x16xf32>) -> tensor<2x2x16x16xf32> {
    %empty = tensor.empty() : tensor<2x8x16x16xf16>
    %lhs_transpose = linalg.transpose ins(%arg0: tensor<8x2x16x16xf16>) outs(%empty: tensor<2x8x16x16xf16>) permutation = [1, 0, 2, 3]
    %mm = linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
      ins(%lhs_transpose, %arg1 : tensor<2x8x16x16xf16>, tensor<8x2x16x16xf16>)
      outs(%arg2 : tensor<2x2x16x16xf32>)
    attrs =  {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>}>} {
    ^bb0(%in: f16, %in_2: f16, %out: f32):
      %4 = arith.extf %in : f16 to f32
      %5 = arith.extf %in_2 : f16 to f32
      %6 = arith.mulf %4, %5 : f32
      %7 = arith.addf %out, %6 : f32
      linalg.yield %7 : f32
    } -> tensor<2x2x16x16xf32>
    return %mm : tensor<2x2x16x16xf32>
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @matmul_16x16x16
//       CHECK:   scf.forall
//       CHECK:     %[[LHS_T:.+]] = linalg.transpose ins({{.*}}: tensor<2x8x1x4xf16>)
//       CHECK:     iree_gpu.multi_mma %[[LHS_T]]
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
//  CHECK-SAME:       : tensor<2x8x1x4xf16>, tensor<8x2x1x4xf16> into tensor<2x2x4x1xf32>
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]
