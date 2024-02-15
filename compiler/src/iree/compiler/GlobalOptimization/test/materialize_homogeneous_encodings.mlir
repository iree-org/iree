// RUN: iree-opt --split-input-file --iree-global-opt-materialize-homogeneous-encodings %s | FileCheck %s

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {target_triple = "x86_64-none-elf", cpu_features = "+avx512f"}>
#map = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#device_target_llvm_cpu = #hal.device.target<"llvm-cpu", {executable_targets = [#executable_target_embedded_elf_x86_64_]}>
module attributes {hal.device.targets = [#device_target_llvm_cpu]} {
  func.func @lhs_encoding(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3]>> -> index, index
    %1 = affine.apply #map()[%0#0, %dim]
    %2 = affine.apply #map()[%0#1, %dim_0]
    %padded = tensor.pad %arg0 low[0, 0] high[%1, %2] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %cst : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
    %3 = iree_linalg_ext.set_encoding %padded : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3]>>
    %4 = iree_linalg_ext.unset_encoding %3 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3]>> -> tensor<?x?xf32>
    return %4 : tensor<?x?xf32>
  }
}
// CHECK-LABEL: func.func @lhs_encoding
// CHECK:         tensor.pack
// CHECK:         tensor.unpack

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan", "vulkan-spirv-fb">
#map = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#device_target_vulkan = #hal.device.target<"vulkan", {executable_targets = [#executable_target_vulkan_spirv_fb], legacy_sync}>
module attributes {hal.device.targets = [#device_target_vulkan]} {
  func.func @lhs_encoding(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3]>> -> index, index
    %1 = affine.apply #map()[%0#0, %dim]
    %2 = affine.apply #map()[%0#1, %dim_0]
    %padded = tensor.pad %arg0 low[0, 0] high[%1, %2] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %cst : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
    %3 = iree_linalg_ext.set_encoding %padded : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3]>>
    %4 = iree_linalg_ext.unset_encoding %3 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3]>> -> tensor<?x?xf32>
    return %4 : tensor<?x?xf32>
  }
}

// vulkan uses default materialization patterns which unsets the encodings.
// CHECK-LABEL: func.func @lhs_encoding
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         return %[[ARG0]]

// -----

#map = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {target_triple = "x86_64-none-elf", cpu_features = "+avx512f"}>
#device_target_llvm_cpu = #hal.device.target<"llvm-cpu", {executable_targets = [#executable_target_embedded_elf_x86_64_]}>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan", "vulkan-spirv-fb">
#device_target_vulkan = #hal.device.target<"vulkan", {executable_targets = [#executable_target_vulkan_spirv_fb], legacy_sync}>
module attributes {hal.device.targets = [#device_target_vulkan, #device_target_llvm_cpu]} {
  func.func @lhs_encoding(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3]>> -> index, index
    %1 = affine.apply #map()[%0#0, %dim]
    %2 = affine.apply #map()[%0#1, %dim_0]
    %padded = tensor.pad %arg0 low[0, 0] high[%1, %2] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %cst : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
    %3 = iree_linalg_ext.set_encoding %padded : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3]>>
    %4 = iree_linalg_ext.unset_encoding %3 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3]>> -> tensor<?x?xf32>
    return %4 : tensor<?x?xf32>
  }
}

// Multiple targets are currently unsupported.
// CHECK-LABEL: func.func @lhs_encoding
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         return %[[ARG0]]

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx90a", ukernels = "none"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
#map4 = affine_map<()[s0] -> ((1 ceildiv s0) * s0)>
#map5 = affine_map<()[s0] -> ((1280 ceildiv s0) * s0)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>
#device_target_rocm = #hal.device.target<"rocm", {executable_targets = [#executable_target_rocm_hsaco_fb], legacy_sync}>
module attributes {hal.device.targets = [#device_target_rocm]} {
  hal.executable public @main_dispatch_517 {
    hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
      hal.executable.export public @nested_module_test ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @nested_module_test() {
          %c1280 = arith.constant 1280 : index
          %c1 = arith.constant 1 : index
          %cst = arith.constant 0.000000e+00 : f32
          %c128 = arith.constant 128 : index
          %c0 = arith.constant 0 : index
          %c394240 = arith.constant 394240 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c128) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1280xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280x1280xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c394240) : !flow.dispatch.tensor<writeonly:tensor<1x1280xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1280xf32>> -> tensor<1x1280xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1280, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x1280xf32>> -> tensor<1280x1280xf32>
          %5:2 = iree_linalg_ext.upper_bound_tile_size tensor<1x1280xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], matmul_narrow_M = 1 : index, user_indexing_maps = [#map, #map1, #map2]>> -> index, index
          %6 = affine.apply #map3()[%5#0, %c1]
          %7 = affine.apply #map3()[%5#1, %c1280]
          %padded = tensor.pad %3 low[0, 0] high[%6, %7] {
          ^bb0(%arg0: index, %arg1: index):
            tensor.yield %cst : f32
          } : tensor<1x1280xf32> to tensor<?x?xf32>
          %8 = iree_linalg_ext.set_encoding %padded : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], original_type = tensor<1x1280xf32>, matmul_narrow_M = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>
          %9:2 = iree_linalg_ext.upper_bound_tile_size tensor<1280x1280xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], matmul_narrow_M = 1 : index, user_indexing_maps = [#map, #map1, #map2]>> -> index, index
          %10 = affine.apply #map3()[%9#0, %c1280]
          %11 = affine.apply #map3()[%9#1, %c1280]
          %padded_0 = tensor.pad %4 low[0, 0] high[%10, %11] {
          ^bb0(%arg0: index, %arg1: index):
            tensor.yield %cst : f32
          } : tensor<1280x1280xf32> to tensor<?x?xf32>
          %12 = iree_linalg_ext.set_encoding %padded_0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], original_type = tensor<1280x1280xf32>, matmul_narrow_M = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>
          %13:2 = iree_linalg_ext.upper_bound_tile_size tensor<1x1280xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], matmul_narrow_M = 1 : index, user_indexing_maps = [#map, #map1, #map2]>> -> index, index
          %14 = affine.apply #map4()[%13#0]
          %15 = affine.apply #map5()[%13#1]
          %16 = tensor.empty(%14, %15) : tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<1x1280xf32>, matmul_narrow_M = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>
          %17 = linalg.fill ins(%cst : f32) outs(%16 : tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<1x1280xf32>, matmul_narrow_M = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<1x1280xf32>, matmul_narrow_M = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>
          %18 = linalg.matmul_transpose_b ins(%8, %12 : tensor<?x?xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], original_type = tensor<1x1280xf32>, matmul_narrow_M = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>, tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], original_type = tensor<1280x1280xf32>, matmul_narrow_M = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>) outs(%17 : tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<1x1280xf32>, matmul_narrow_M = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<1x1280xf32>, matmul_narrow_M = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>
          %19 = iree_linalg_ext.unset_encoding %18 : tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<1x1280xf32>, matmul_narrow_M = 1 : index, user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?x?xf32>
          %extracted_slice = tensor.extract_slice %19[0, 0] [1, 1280] [1, 1] : tensor<?x?xf32> to tensor<1x1280xf32>
          flow.dispatch.tensor.store %extracted_slice, %2, offsets = [0, 0], sizes = [1, 1280], strides = [1, 1] : tensor<1x1280xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x1280xf32>>
          return
        }
      }
    }
  }
}

// CHECK-LABEL: func.func @nested_module_test
//   CHECK-NOT: iree_linalg_ext.set_encoding
