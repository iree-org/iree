// RUN: iree-opt --pass-pipeline='builtin.module(iree-rocm-apply-builtin-pdl-patterns{targets=gfx942 enable-tensor-ukernels=true})' \
// RUN:   --mlir-print-local-scope --split-input-file %s | FileCheck %s

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @matmul_f8(%arg0: tensor<1x128x4096xf8E4M3FNUZ>, %arg1: tensor<1024x4096xf8E4M3FNUZ>) -> tensor<1x128x1024xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x128x1024xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x128x1024xf32>) -> tensor<1x128x1024xf32>
  %2 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x128x4096xf8E4M3FNUZ>, tensor<1024x4096xf8E4M3FNUZ>) outs(%1 : tensor<1x128x1024xf32>) {
    ^bb0(%in: f8E4M3FNUZ, %in_4: f8E4M3FNUZ, %out: f32):
      %12 = arith.extf %in : f8E4M3FNUZ to f32
      %13 = arith.extf %in_4 : f8E4M3FNUZ to f32
      %14 = arith.mulf %12, %13 : f32
      %15 = arith.addf %out, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<1x128x1024xf32>
  return %2 : tensor<1x128x1024xf32>
}
// CHECK-LABEL: @matmul_f8
// CHECK:         linalg.generic
// CHECK-SAME:      compilation_info = #iree_codegen.compilation_info
// CHECK-SAME:      lowering_config =
// CHECK-SAME:      translation_info =
// CHECK-SAME:      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_medium_f8_expanded", tensor>
