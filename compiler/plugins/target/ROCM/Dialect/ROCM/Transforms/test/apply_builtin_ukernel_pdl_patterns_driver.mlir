// RUN: iree-opt --pass-pipeline='builtin.module(iree-rocm-apply-builtin-pdl-patterns-driver{enable-tensor-ukernels=true})' \
// RUN:   --mlir-print-local-scope --split-input-file %s | FileCheck %s

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
}
// CHECK-LABEL: util.func private @pingpong_medium_f8_expanded
// CHECK:         iree_codegen.inner_tiled

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
// CHECK-SAME:      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_large_f8_expanded", tensor>
// CHECK-LABEL: util.func private @pingpong_large_f8_expanded
// CHECK:         iree_codegen.inner_tiled

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
  func.func @matmul_f16_medium_expanded(%arg0: tensor<1x128x4096xf16>, %arg1: tensor<1024x4096xf16>) -> tensor<1x128x1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x128x1024xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x128x1024xf32>) -> tensor<1x128x1024xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x128x4096xf16>, tensor<1024x4096xf16>) outs(%1 : tensor<1x128x1024xf32>) {
      ^bb0(%in: f16, %in_4: f16, %out: f32):
        %12 = arith.extf %in : f16 to f32
        %13 = arith.extf %in_4 : f16 to f32
        %14 = arith.mulf %12, %13 : f32
        %15 = arith.addf %out, %14 : f32
        linalg.yield %15 : f32
      } -> tensor<1x128x1024xf32>
    return %2 : tensor<1x128x1024xf32>
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
