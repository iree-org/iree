// RUN: iree-opt --split-input-file --iree-hal-device-assignment-pipeline --iree-global-opt-materialize-homogeneous-encodings %s | FileCheck %s

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb">
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 16, 16, 16>>
#device_target_vulkan = #hal.device.target<"vulkan", [#executable_target_vulkan_spirv_fb]> : !hal.device
module attributes {hal.device.targets = [#device_target_vulkan]} {
  util.func public @lhs_encoding(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %3 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
    %4 = iree_encoding.unset_encoding %3 : tensor<?x?xf32, #encoding> -> tensor<?x?xf32>{%d0, %d1}
    util.return %4 : tensor<?x?xf32>
  }
}

// Vulkan uses default materialization patterns which unsets the encodings.
// CHECK-LABEL: util.func public @lhs_encoding
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         util.return %[[ARG0]]

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 16, 16, 16>>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {target_triple = "x86_64-none-elf", cpu_features = "+avx512f"}>
#device_target_llvm_cpu = #hal.device.target<"local", [#executable_target_embedded_elf_x86_64_]> : !hal.device
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb">
#device_target_vulkan = #hal.device.target<"vulkan", [#executable_target_vulkan_spirv_fb]> : !hal.device
module attributes {hal.device.targets = [#hal.device.select<[#device_target_vulkan, #device_target_llvm_cpu]> : !hal.device]} {
  util.func public @lhs_encoding(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %3 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
    %4 = iree_encoding.unset_encoding %3 : tensor<?x?xf32, #encoding> -> tensor<?x?xf32>{%d0, %d1}
    util.return %4 : tensor<?x?xf32>
  }
}

// Multiple targets are currently unsupported.
// CHECK-LABEL: util.func public @lhs_encoding
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         util.return %[[ARG0]]
