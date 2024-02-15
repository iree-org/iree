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

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan", "vulkan-spirv-fb">
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
#device_target_vulkan = #hal.device.target<"vulkan", {executable_targets = [#executable_target_vulkan_spirv_fb], legacy_sync}>
module {
  module attributes {hal.device.targets = [#device_target_vulkan]} {
    func.func @nested_lhs_encoding(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
      %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
      %0:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index
      %1 = affine.apply #map3()[%0#0, %dim]
      %2 = affine.apply #map3()[%0#1, %dim_0]
      %padded = tensor.pad %arg0 low[0, 0] high[%1, %2] {
      ^bb0(%arg1: index, %arg2: index):
        tensor.yield %cst : f32
      } : tensor<?x?xf32> to tensor<?x?xf32>
      %3 = iree_linalg_ext.set_encoding %padded : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
      %4 = iree_linalg_ext.unset_encoding %3 : tensor<?x?xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?x?xf32>
      return %4 : tensor<?x?xf32>
    }
  }
}

// CHECK-LABEL: func.func @nested_lhs_encoding
//   CHECK-NOT: iree_linalg_ext.set_encoding
