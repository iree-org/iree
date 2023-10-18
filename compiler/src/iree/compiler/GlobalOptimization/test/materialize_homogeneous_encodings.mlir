// RUN: iree-opt --split-input-file --iree-global-opt-materialize-homogeneous-encodings %s | FileCheck %s

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f"}>
#map = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
#device_target_llvm_cpu = #hal.device.target<"llvm-cpu", {executable_targets = [#executable_target_embedded_elf_x86_64_]}>
module attributes {hal.device.targets = [#device_target_llvm_cpu]} {
  func.func @lhs_encoding(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>> -> index, index
    %1 = affine.apply #map()[%0#0, %dim]
    %2 = affine.apply #map()[%0#1, %dim_0]
    %padded = tensor.pad %arg0 low[0, 0] high[%1, %2] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %cst : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
    %3 = iree_linalg_ext.set_encoding %padded : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
    %4 = iree_linalg_ext.unset_encoding %3 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
    return %4 : tensor<?x?xf32>
  }
}
// CHECK-LABEL: func.func @lhs_encoding
// CHECK:         tensor.pack
// CHECK:         tensor.unpack

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan", "vulkan-spirv-fb">
#map = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
#device_target_vulkan = #hal.device.target<"vulkan", {executable_targets = [#executable_target_vulkan_spirv_fb], legacy_sync}>
module attributes {hal.device.targets = [#device_target_vulkan]} {
  func.func @lhs_encoding(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0:2 = iree_linalg_ext.upper_bound_tile_size tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>> -> index, index
    %1 = affine.apply #map()[%0#0, %dim]
    %2 = affine.apply #map()[%0#1, %dim_0]
    %padded = tensor.pad %arg0 low[0, 0] high[%1, %2] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %cst : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
    %3 = iree_linalg_ext.set_encoding %padded : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
    %4 = iree_linalg_ext.unset_encoding %3 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
    return %4 : tensor<?x?xf32>
  }
}

// vulkan does not implement buildMaterializeEncodingsPassPipeline method.
// CHECK-LABEL: func.func @lhs_encoding
// CHECK:         iree_linalg_ext.upper_bound_tile_size
// CHECK:         iree_linalg_ext.set_encoding
