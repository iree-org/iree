// RUN: iree-opt --pass-pipeline='builtin.module(iree-rocm-apply-builtin-pdl-patterns-driver{enable-tensor-ukernels=true})' \
// RUN:   --mlir-print-local-scope --split-input-file %s | FileCheck %s

// Test remarks output for ukernels
// RUN: iree-opt --pass-pipeline='builtin.module(iree-rocm-apply-builtin-pdl-patterns-driver{enable-tensor-ukernels=true})' \
// RUN:   --remarks-filter=".*" --split-input-file %s 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "",
                                               wgp = <compute = fp16, storage =  b16,
                                               subgroup =  none,
                                               subgroup_size_choices = [64],
                                               max_workgroup_sizes = [1024, 1024, 1024],
                                               max_thread_count_per_workgroup = 1024,
                                               max_workgroup_memory_bytes = 65536,
                                               max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>,
   ukernels = "none"}>
module attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
} {
  func.func @matmul_f8_medium_expanded(%arg0: tensor<1x128x4096xf8E4M3FNUZ>, %arg1: tensor<1024x4096xf8E4M3FNUZ>) -> tensor<1x128x1024xf32> {
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
  // Check that a second function requiring the same ukernel doesn't lead to a 'redefinition of symbol named ...' error.
  func.func @matmul_f8_medium_expanded_2(%arg0: tensor<1x128x4096xf8E4M3FNUZ>, %arg1: tensor<1024x4096xf8E4M3FNUZ>) -> tensor<1x128x1024xf32> {
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
}
// CHECK-LABEL: util.func private @pingpong_medium_f8E4M3FNUZ_expanded
// CHECK:         iree_codegen.inner_tiled

// CHECK-REMARKS:      [Analysis] UKernel
// CHECK-REMARKS-SAME:   Category:ApplyBuiltinPDLPatternsDriverPass
// CHECK-REMARKS-SAME:   Remark=pingpong_medium_f8E4M3FNUZ_expanded
// CHECK-REMARKS:      [Analysis] UKernel
// CHECK-REMARKS-SAME:   Category:ApplyBuiltinPDLPatternsDriverPass
// CHECK-REMARKS-SAME:   Remark=pingpong_medium_f8E4M3FNUZ_expanded

// -----

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "",
                                               wgp = <compute = fp16, storage =  b16,
                                               subgroup =  none,
                                               subgroup_size_choices = [64],
                                               max_workgroup_sizes = [1024, 1024, 1024],
                                               max_thread_count_per_workgroup = 1024,
                                               max_workgroup_memory_bytes = 65536,
                                               max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>,
   ukernels = "none"}>
module attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
} {
  func.func @matmul_f8_large_expanded(%arg0: tensor<1x256x4096xf8E4M3FNUZ>, %arg1: tensor<1024x4096xf8E4M3FNUZ>) -> tensor<1x256x1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x256x1024xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x256x1024xf32>) -> tensor<1x256x1024xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x256x4096xf8E4M3FNUZ>, tensor<1024x4096xf8E4M3FNUZ>) outs(%1 : tensor<1x256x1024xf32>) {
      ^bb0(%in: f8E4M3FNUZ, %in_4: f8E4M3FNUZ, %out: f32):
        %12 = arith.extf %in : f8E4M3FNUZ to f32
        %13 = arith.extf %in_4 : f8E4M3FNUZ to f32
        %14 = arith.mulf %12, %13 : f32
        %15 = arith.addf %out, %14 : f32
        linalg.yield %15 : f32
      } -> tensor<1x256x1024xf32>
    return %2 : tensor<1x256x1024xf32>
  }
}
// CHECK-LABEL: @matmul_f8_large_expanded
// CHECK:         linalg.generic
// CHECK-SAME:      compilation_info = #iree_codegen.compilation_info
// CHECK-SAME:      lowering_config =
// CHECK-SAME:      translation_info =
// CHECK-SAME:      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_large_f8E4M3FNUZ_expanded", tensor>
// CHECK-LABEL: util.func private @pingpong_large_f8E4M3FNUZ_expanded
// CHECK:         iree_codegen.inner_tiled

// CHECK-REMARKS:      [Analysis] UKernel
// CHECK-REMARKS-SAME:   Category:ApplyBuiltinPDLPatternsDriverPass
// CHECK-REMARKS-SAME:   Remark=pingpong_large_f8E4M3FNUZ_expanded

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "",
                                               wgp = <compute = fp16, storage =  b16,
                                               subgroup =  none,
                                               subgroup_size_choices = [64],
                                               max_workgroup_sizes = [1024, 1024, 1024],
                                               max_thread_count_per_workgroup = 1024,
                                               max_workgroup_memory_bytes = 65536,
                                               max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>,
   ukernels = "none"}>
module attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
} {
  func.func @matmul_f16_large(%arg0: tensor<1024x4096xf16>, %arg1: tensor<1024x4096xf16>) -> tensor<1024x1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1024x1024xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1024x4096xf16>, tensor<1024x4096xf16>) outs(%1 : tensor<1024x1024xf32>) {
      ^bb0(%in: f16, %in_4: f16, %out: f32):
        %12 = arith.extf %in : f16 to f32
        %13 = arith.extf %in_4 : f16 to f32
        %14 = arith.mulf %12, %13 : f32
        %15 = arith.addf %out, %14 : f32
        linalg.yield %15 : f32
      } -> tensor<1024x1024xf32>
    return %2 : tensor<1024x1024xf32>
  }
}
// CHECK-LABEL: @matmul_f16_large
// CHECK:         linalg.generic
// CHECK-SAME:      compilation_info = #iree_codegen.compilation_info
// CHECK-SAME:      lowering_config =
// CHECK-SAME:      translation_info =
// CHECK-SAME:      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_large_f16", tensor>
// CHECK-LABEL: util.func private @pingpong_large_f16
// CHECK:         iree_codegen.inner_tiled

// CHECK-REMARKS:      [Analysis] UKernel
// CHECK-REMARKS-SAME:   Category:ApplyBuiltinPDLPatternsDriverPass
// CHECK-REMARKS-SAME:   Remark=pingpong_large_f16

// -----

// Test that the no ukernel is selected if Batch * M is too small.

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "",
                                               wgp = <compute = fp16, storage =  b16,
                                               subgroup =  none,
                                               subgroup_size_choices = [64],
                                               max_workgroup_sizes = [1024, 1024, 1024],
                                               max_thread_count_per_workgroup = 1024,
                                               max_workgroup_memory_bytes = 65536,
                                               max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>,
   ukernels = "none"}>
module attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
} {
  func.func @negative_matmul_f16_medium_expanded(%arg0: tensor<7x128x4096xf16>, %arg1: tensor<1024x4096xf16>) -> tensor<7x128x1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<7x128x1024xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<7x128x1024xf32>) -> tensor<7x128x1024xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<7x128x4096xf16>, tensor<1024x4096xf16>) outs(%1 : tensor<7x128x1024xf32>) {
      ^bb0(%in: f16, %in_4: f16, %out: f32):
        %12 = arith.extf %in : f16 to f32
        %13 = arith.extf %in_4 : f16 to f32
        %14 = arith.mulf %12, %13 : f32
        %15 = arith.addf %out, %14 : f32
        linalg.yield %15 : f32
      } -> tensor<7x128x1024xf32>
    return %2 : tensor<7x128x1024xf32>
  }
}
// CHECK-LABEL: @negative_matmul_f16_medium_expanded
// CHECK-NOT:     compilation_info = #iree_codegen.compilation_info
// CHECK-NOT:     iree_codegen.ukernel
// CHECK-NOT:     iree_codegen.inner_tiled

// -----

// Test that the medium ukernel is selected once Batch * M becomes large enough.

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "",
                                               wgp = <compute = fp16, storage =  b16,
                                               subgroup =  none,
                                               subgroup_size_choices = [64],
                                               max_workgroup_sizes = [1024, 1024, 1024],
                                               max_thread_count_per_workgroup = 1024,
                                               max_workgroup_memory_bytes = 65536,
                                               max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>,
   ukernels = "none"}>
module attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
} {
  func.func @matmul_f16_medium_expanded(%arg0: tensor<8x128x4096xf16>, %arg1: tensor<1024x4096xf16>) -> tensor<8x128x1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8x128x1024xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8x128x1024xf32>) -> tensor<8x128x1024xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x128x4096xf16>, tensor<1024x4096xf16>) outs(%1 : tensor<8x128x1024xf32>) {
      ^bb0(%in: f16, %in_4: f16, %out: f32):
        %12 = arith.extf %in : f16 to f32
        %13 = arith.extf %in_4 : f16 to f32
        %14 = arith.mulf %12, %13 : f32
        %15 = arith.addf %out, %14 : f32
        linalg.yield %15 : f32
      } -> tensor<8x128x1024xf32>
    return %2 : tensor<8x128x1024xf32>
  }
}
// CHECK-LABEL: @matmul_f16_medium_expanded
// CHECK:         linalg.generic
// CHECK-SAME:      compilation_info = #iree_codegen.compilation_info
// CHECK-SAME:      lowering_config =
// CHECK-SAME:      translation_info =
// CHECK-SAME:      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_medium_f16_expanded", tensor>
// CHECK-LABEL: util.func private @pingpong_medium_f16_expanded
// CHECK:         iree_codegen.inner_tiled

// CHECK-REMARKS:      [Analysis] UKernel
// CHECK-REMARKS-SAME:   Category:ApplyBuiltinPDLPatternsDriverPass
// CHECK-REMARKS-SAME:   Remark=pingpong_medium_f16_expanded

// -----

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "",
                                               wgp = <compute = fp16, storage =  b16,
                                               subgroup =  none,
                                               subgroup_size_choices = [64],
                                               max_workgroup_sizes = [1024, 1024, 1024],
                                               max_thread_count_per_workgroup = 1024,
                                               max_workgroup_memory_bytes = 65536,
                                               max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>,
   ukernels = "none"}>
module attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
} {
  func.func @matmul_f16_large_expanded(%arg0: tensor<1x256x4096xf16>, %arg1: tensor<1024x4096xf16>) -> tensor<1x256x1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x256x1024xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x256x1024xf32>) -> tensor<1x256x1024xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x256x4096xf16>, tensor<1024x4096xf16>) outs(%1 : tensor<1x256x1024xf32>) {
      ^bb0(%in: f16, %in_4: f16, %out: f32):
        %12 = arith.extf %in : f16 to f32
        %13 = arith.extf %in_4 : f16 to f32
        %14 = arith.mulf %12, %13 : f32
        %15 = arith.addf %out, %14 : f32
        linalg.yield %15 : f32
      } -> tensor<1x256x1024xf32>
    return %2 : tensor<1x256x1024xf32>
  }
}
// CHECK-LABEL: @matmul_f16_large_expanded
// CHECK:         linalg.generic
// CHECK-SAME:      compilation_info = #iree_codegen.compilation_info
// CHECK-SAME:      lowering_config =
// CHECK-SAME:      translation_info =
// CHECK-SAME:      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_large_f16_expanded", tensor>
// CHECK-LABEL: util.func private @pingpong_large_f16_expanded
// CHECK:         iree_codegen.inner_tiled

// CHECK-REMARKS:      [Analysis] UKernel
// CHECK-REMARKS-SAME:   Category:ApplyBuiltinPDLPatternsDriverPass
// CHECK-REMARKS-SAME:   Remark=pingpong_large_f16_expanded

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "",
                                               wgp = <compute = fp16, storage =  b16,
                                               subgroup =  none,
                                               subgroup_size_choices = [64],
                                               max_workgroup_sizes = [1024, 1024, 1024],
                                               max_thread_count_per_workgroup = 1024,
                                               max_workgroup_memory_bytes = 65536,
                                               max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>,
   ukernels = "none"}>
module attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
} {
  func.func @matmul_bf16_large(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<1024x4096xbf16>) -> tensor<1024x1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1024x1024xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1024x4096xbf16>, tensor<1024x4096xbf16>) outs(%1 : tensor<1024x1024xf32>) {
      ^bb0(%in: bf16, %in_4: bf16, %out: f32):
        %12 = arith.extf %in : bf16 to f32
        %13 = arith.extf %in_4 : bf16 to f32
        %14 = arith.mulf %12, %13 : f32
        %15 = arith.addf %out, %14 : f32
        linalg.yield %15 : f32
      } -> tensor<1024x1024xf32>
    return %2 : tensor<1024x1024xf32>
  }
}
// CHECK-LABEL: @matmul_bf16_large
// CHECK:         linalg.generic
// CHECK-SAME:      compilation_info = #iree_codegen.compilation_info
// CHECK-SAME:      lowering_config =
// CHECK-SAME:      translation_info =
// CHECK-SAME:      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_large_bf16", tensor>
// CHECK-LABEL: util.func private @pingpong_large_bf16
// CHECK:         iree_codegen.inner_tiled

// CHECK-REMARKS:      [Analysis] UKernel
// CHECK-REMARKS-SAME:   Category:ApplyBuiltinPDLPatternsDriverPass
// CHECK-REMARKS-SAME:   Remark=pingpong_large_bf16

// -----

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "",
                                               wgp = <compute = fp16, storage =  b16,
                                               subgroup =  none,
                                               subgroup_size_choices = [64],
                                               max_workgroup_sizes = [1024, 1024, 1024],
                                               max_thread_count_per_workgroup = 1024,
                                               max_workgroup_memory_bytes = 65536,
                                               max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>,
   ukernels = "none"}>
module attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
} {
  func.func @matmul_bf16_expanded_large(%arg0: tensor<1x256x4096xbf16>, %arg1: tensor<1024x4096xbf16>) -> tensor<1x256x1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x256x1024xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x256x1024xf32>) -> tensor<1x256x1024xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x256x4096xbf16>, tensor<1024x4096xbf16>) outs(%1 : tensor<1x256x1024xf32>) {
      ^bb0(%in: bf16, %in_4: bf16, %out: f32):
        %12 = arith.extf %in : bf16 to f32
        %13 = arith.extf %in_4 : bf16 to f32
        %14 = arith.mulf %12, %13 : f32
        %15 = arith.addf %out, %14 : f32
        linalg.yield %15 : f32
      } -> tensor<1x256x1024xf32>
    return %2 : tensor<1x256x1024xf32>
  }
}
// CHECK-LABEL: @matmul_bf16_expanded_large
// CHECK:         linalg.generic
// CHECK-SAME:      compilation_info = #iree_codegen.compilation_info
// CHECK-SAME:      lowering_config =
// CHECK-SAME:      translation_info =
// CHECK-SAME:      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_large_bf16_expanded", tensor>
// CHECK-LABEL: util.func private @pingpong_large_bf16_expanded
// CHECK:         iree_codegen.inner_tiled

// CHECK-REMARKS:      [Analysis] UKernel
// CHECK-REMARKS-SAME:   Category:ApplyBuiltinPDLPatternsDriverPass
// CHECK-REMARKS-SAME:   Remark=pingpong_large_bf16_expanded

// -----

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "",
                                               wgp = <compute = fp16, storage =  b16,
                                               subgroup =  none,
                                               subgroup_size_choices = [64],
                                               max_workgroup_sizes = [1024, 1024, 1024],
                                               max_thread_count_per_workgroup = 1024,
                                               max_workgroup_memory_bytes = 65536,
                                               max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>,
   ukernels = "none"}>
module attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
} {
  func.func @matmul_bf16_expanded_medium(%arg0: tensor<1x128x4096xbf16>, %arg1: tensor<1024x4096xbf16>) -> tensor<1x128x1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x128x1024xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x128x1024xf32>) -> tensor<1x128x1024xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x128x4096xbf16>, tensor<1024x4096xbf16>) outs(%1 : tensor<1x128x1024xf32>) {
      ^bb0(%in: bf16, %in_4: bf16, %out: f32):
        %12 = arith.extf %in : bf16 to f32
        %13 = arith.extf %in_4 : bf16 to f32
        %14 = arith.mulf %12, %13 : f32
        %15 = arith.addf %out, %14 : f32
        linalg.yield %15 : f32
      } -> tensor<1x128x1024xf32>
    return %2 : tensor<1x128x1024xf32>
  }
}
// CHECK-LABEL: @matmul_bf16_expanded_medium
// CHECK:         linalg.generic
// CHECK-SAME:      compilation_info = #iree_codegen.compilation_info
// CHECK-SAME:      lowering_config =
// CHECK-SAME:      translation_info =
// CHECK-SAME:      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_medium_bf16_expanded", tensor>
// CHECK-LABEL: util.func private @pingpong_medium_bf16_expanded
// CHECK:         iree_codegen.inner_tiled

// CHECK-REMARKS:      [Analysis] UKernel
// CHECK-REMARKS-SAME:   Category:ApplyBuiltinPDLPatternsDriverPass
// CHECK-REMARKS-SAME:   Remark=pingpong_medium_bf16_expanded

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "",
                                               wgp = <compute = fp16, storage =  b16,
                                               subgroup =  none,
                                               subgroup_size_choices = [64],
                                               max_workgroup_sizes = [1024, 1024, 1024],
                                               max_thread_count_per_workgroup = 1024,
                                               max_workgroup_memory_bytes = 65536,
                                               max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>,
   ukernels = "none"}>
module attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
} {
  func.func @inner_tiled_f8_large(%arg0: tensor<1x128x2x8x4x16x8xf8E4M3FNUZ>, %arg1: tensor<16x128x4x4x4x16x8xf8E4M3FNUZ>) -> tensor<1x16x2x4x8x4x4x16x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x16x2x4x8x4x4x16x4xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x16x2x4x8x4x4x16x4xf32>) -> tensor<1x16x2x4x8x4x4x16x4xf32>
    %2 = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%1){
          indexing_maps = [#map1, #map2, #map3],
          iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
          kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x32_F8E4M3FNUZ, intrinsics_m = 8, subgroups_m = 2, intrinsics_n = 4, subgroups_n = 4>,
          semantics = #iree_gpu.mma_semantics<distributed = false, opaque = false>
        } : tensor<1x128x2x8x4x16x8xf8E4M3FNUZ>, tensor<16x128x4x4x4x16x8xf8E4M3FNUZ> into tensor<1x16x2x4x8x4x4x16x4xf32>
    return %2 : tensor<1x16x2x4x8x4x4x16x4xf32>
  }
}
// CHECK-LABEL: @inner_tiled_f8_large
// CHECK:         iree_codegen.inner_tiled
// CHECK-SAME:      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_dt_large_f8E4M3FNUZ", tensor>

// CHECK-REMARKS:      [Analysis] UKernel
// CHECK-REMARKS-SAME:   Category:ApplyBuiltinPDLPatternsDriverPass
// CHECK-REMARKS-SAME:   Remark=pingpong_dt_large_f8E4M3FNUZ

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "",
                                               wgp = <compute = fp16, storage =  b16,
                                               subgroup =  none,
                                               subgroup_size_choices = [64],
                                               max_workgroup_sizes = [1024, 1024, 1024],
                                               max_thread_count_per_workgroup = 1024,
                                               max_workgroup_memory_bytes = 65536,
                                               max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>,
   ukernels = "none"}>
module attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
} {
  func.func @inner_tiled_f8_medium(%arg0: tensor<1x64x8x4x16x2x8xf8E4M3FNUZ>, %arg1: tensor<4x64x8x2x4x16x2x8xf8E4M3FNUZ>) -> tensor<1x4x8x8x2x4x16x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x4x8x8x2x4x16x4xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x4x8x8x2x4x16x4xf32>) -> tensor<1x4x8x8x2x4x16x4xf32>
    %2 = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%1){
          indexing_maps = [#map1, #map2, #map3],
          iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
          kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x32_F8E4M3FNUZ,  intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 8, intrinsics_k = 2, operands_interleaving_intrinsics_k = [0, 1]>,
          semantics = #iree_gpu.mma_semantics<distributed = false, opaque = false>
        } : tensor<1x64x8x4x16x2x8xf8E4M3FNUZ>, tensor<4x64x8x2x4x16x2x8xf8E4M3FNUZ> into tensor<1x4x8x8x2x4x16x4xf32>
    return %2 : tensor<1x4x8x8x2x4x16x4xf32>
  }
}
// CHECK-LABEL: @inner_tiled_f8_medium
// CHECK:         iree_codegen.inner_tiled
// CHECK-SAME:      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_dt_medium_f8E4M3FNUZ", tensor>

// CHECK-REMARKS:      [Analysis] UKernel
// CHECK-REMARKS-SAME:   Category:ApplyBuiltinPDLPatternsDriverPass
// CHECK-REMARKS-SAME:   Remark=pingpong_dt_medium_f8E4M3FNUZ

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "",
                                               wgp = <compute = fp16, storage =  b16,
                                               subgroup =  none,
                                               subgroup_size_choices = [64],
                                               max_workgroup_sizes = [1024, 1024, 1024],
                                               max_thread_count_per_workgroup = 1024,
                                               max_workgroup_memory_bytes = 65536,
                                               max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>,
   ukernels = "none"}>
module attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
} {
  func.func @inner_tiled_f16_large(%arg0: tensor<1x256x2x8x4x16x4xf16>, %arg1: tensor<501x256x4x4x4x16x4xf16>) -> tensor<1x501x2x4x8x4x4x16x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x501x2x4x8x4x4x16x4xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x501x2x4x8x4x4x16x4xf32>) -> tensor<1x501x2x4x8x4x4x16x4xf32>
    %2 = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%1){
          indexing_maps = [#map1, #map2, #map3],
          iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
          kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x16_F16, intrinsics_m = 8, subgroups_m = 2, intrinsics_n = 4, subgroups_n = 4>,
          semantics = #iree_gpu.mma_semantics<distributed = false, opaque = false>
        } : tensor<1x256x2x8x4x16x4xf16>, tensor<501x256x4x4x4x16x4xf16> into tensor<1x501x2x4x8x4x4x16x4xf32>
    return %2 : tensor<1x501x2x4x8x4x4x16x4xf32>
  }
}
// CHECK-LABEL: @inner_tiled_f16_large
// CHECK:         iree_codegen.inner_tiled
// CHECK-SAME:      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_dt_large_f16", tensor>

// CHECK-REMARKS:      [Analysis] UKernel
// CHECK-REMARKS-SAME:   Category:ApplyBuiltinPDLPatternsDriverPass
// CHECK-REMARKS-SAME:   Remark=pingpong_dt_large_f16
