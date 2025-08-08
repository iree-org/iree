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
}
// CHECK-LABEL: util.func private @pingpong_medium_f8_expanded
// CHECK:         iree_codegen.inner_tiled
