// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-rocm-apply-builtin-pdl-patterns{targets=gfx942 enable-tensor-ukernels=true}))' \
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

// -----

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @negative_matmul_f8(%arg0: tensor<1x128x256xf8E4M3FNUZ>, %arg1: tensor<1024x256xf8E4M3FNUZ>) -> tensor<1x128x1024xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x128x1024xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x128x1024xf32>) -> tensor<1x128x1024xf32>
  %2 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x128x256xf8E4M3FNUZ>, tensor<1024x256xf8E4M3FNUZ>) outs(%1 : tensor<1x128x1024xf32>) {
    ^bb0(%in: f8E4M3FNUZ, %in_4: f8E4M3FNUZ, %out: f32):
      %12 = arith.extf %in : f8E4M3FNUZ to f32
      %13 = arith.extf %in_4 : f8E4M3FNUZ to f32
      %14 = arith.mulf %12, %13 : f32
      %15 = arith.addf %out, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<1x128x1024xf32>
  return %2 : tensor<1x128x1024xf32>
}
// CHECK-LABEL: @negative_matmul_f8
// CHECK-NOT:     compilation_info = #iree_codegen.compilation_info
// CHECK-NOT:     iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_medium_f8_expanded", tensor>

// -----

// Through a constraint, the inner dimension is known to be a multiple of 128 and has a lower bound of 512, so should be matched.

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @matmul_f8_dynamic(%arg0: index) -> tensor<1x128x1024xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = util.assume.int %arg0<umin = 512, udiv = 128> : index
  %1 = tensor.empty(%0) : tensor<1x128x?xf8E4M3FNUZ>
  %2 = tensor.empty(%0) : tensor<1024x?xf8E4M3FNUZ>
  %3 = tensor.empty() : tensor<1x128x1024xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x128x1024xf32>) -> tensor<1x128x1024xf32>
  %5 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1, %2 : tensor<1x128x?xf8E4M3FNUZ>, tensor<1024x?xf8E4M3FNUZ>) outs(%4 : tensor<1x128x1024xf32>) {
  ^bb0(%in: f8E4M3FNUZ, %in_0: f8E4M3FNUZ, %out: f32):
    %6 = arith.extf %in : f8E4M3FNUZ to f32
    %7 = arith.extf %in_0 : f8E4M3FNUZ to f32
    %8 = arith.mulf %6, %7 : f32
    %9 = arith.addf %out, %8 : f32
    linalg.yield %9 : f32
  } -> tensor<1x128x1024xf32>
  return %5 : tensor<1x128x1024xf32>
}
// CHECK-LABEL: @matmul_f8_dynamic
// CHECK:         linalg.generic
// CHECK-SAME:      compilation_info = #iree_codegen.compilation_info
// CHECK-SAME:      lowering_config =
// CHECK-SAME:      translation_info =
// CHECK-SAME:      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_medium_f8_expanded", tensor>

// -----

// The dynamic dimension is not a multiple of 128.

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @negative_matmul_f8_dynamic_multiple_of(%arg0: tensor<1024x512xf8E4M3FNUZ>, %arg1: index) -> tensor<1x?x1024xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = util.assume.int %arg1<udiv = 64> : index
  %1 = tensor.empty(%0) : tensor<1x?x512xf8E4M3FNUZ>
  %2 = tensor.empty(%0) : tensor<1x?x1024xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x?x1024xf32>) -> tensor<1x?x1024xf32>
  %4 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1, %arg0 : tensor<1x?x512xf8E4M3FNUZ>, tensor<1024x512xf8E4M3FNUZ>) outs(%3 : tensor<1x?x1024xf32>) {
  ^bb0(%in: f8E4M3FNUZ, %in_0: f8E4M3FNUZ, %out: f32):
    %5 = arith.extf %in : f8E4M3FNUZ to f32
    %6 = arith.extf %in_0 : f8E4M3FNUZ to f32
    %7 = arith.mulf %5, %6 : f32
    %8 = arith.addf %out, %7 : f32
    linalg.yield %8 : f32
  } -> tensor<1x?x1024xf32>
  return %4 : tensor<1x?x1024xf32>
}
// CHECK-LABEL: @negative_matmul_f8_dynamic_multiple_of
// CHECK-NOT:     compilation_info = #iree_codegen.compilation_info
// CHECK-NOT:     iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_medium_f8_expanded", tensor>

// -----

// The dynamic dimension is a multiple of 128, but doesn't have a lower bound of 512.

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @negative_matmul_f8_dynamic_lower_bound(%arg0: index) -> tensor<1x128x1024xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = util.assume.int %arg0<umin = 256, udiv = 128> : index
  %1 = tensor.empty(%0) : tensor<1x128x?xf8E4M3FNUZ>
  %2 = tensor.empty(%0) : tensor<1024x?xf8E4M3FNUZ>
  %3 = tensor.empty() : tensor<1x128x1024xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x128x1024xf32>) -> tensor<1x128x1024xf32>
  %5 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1, %2 : tensor<1x128x?xf8E4M3FNUZ>, tensor<1024x?xf8E4M3FNUZ>) outs(%4 : tensor<1x128x1024xf32>) {
  ^bb0(%in: f8E4M3FNUZ, %in_0: f8E4M3FNUZ, %out: f32):
    %6 = arith.extf %in : f8E4M3FNUZ to f32
    %7 = arith.extf %in_0 : f8E4M3FNUZ to f32
    %8 = arith.mulf %6, %7 : f32
    %9 = arith.addf %out, %8 : f32
    linalg.yield %9 : f32
  } -> tensor<1x128x1024xf32>
  return %5 : tensor<1x128x1024xf32>
}
// CHECK-LABEL: @negative_matmul_f8_dynamic_lower_bound
// CHECK-NOT:     compilation_info = #iree_codegen.compilation_info
// CHECK-NOT:     iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_medium_f8_expanded", tensor>
