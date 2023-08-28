// RUN: iree-opt --split-input-file --iree-global-opt-materialize-homogeneous-encodings %s | FileCheck %s

#map = affine_map<()[s0] -> ((1025 ceildiv s0) * s0 - 1025)>
#map1 = affine_map<()[s0] -> ((257 ceildiv s0) * s0 - 257)>
#map2 = affine_map<()[s0] -> ((513 ceildiv s0) * s0 - 513)>
#map3 = affine_map<()[s0] -> ((1025 ceildiv s0) * s0)>
#map4 = affine_map<()[s0] -> ((513 ceildiv s0) * s0)>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "cascadelake", cpu_features = "+cmov,+mmx,+popcnt,+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+avx,+avx2,+fma,+avx512f,+bmi,+bmi2,+aes,+pclmul,+avx512vl,+avx512bw,+avx512dq,+avx512cd,+avx512vnni,+adx,+clflushopt,+clwb,+cx16,+cx8,+crc32,+f16c,+fsgsbase,+fxsr,+invpcid,+lzcnt,+movbe,+pku,+prfchw,+rdrnd,+rdseed,+sahf,+x87,+xsave,+xsavec,+xsaveopt,+xsaves", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-unknown-unknown-eabi-elf", ukernels = true}>
#device_target_llvm_cpu = #hal.device.target<"llvm-cpu", {executable_targets = [#executable_target_embedded_elf_x86_64_]}>
module attributes {hal.device.targets = [#device_target_llvm_cpu]} {
  func.func @matmul_1025x513x257(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<257x513xf32>
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<1025x257xf32>
    %1:2 = iree_linalg_ext.upper_bound_tile_size tensor<1025x257xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS>> -> index, index
    %2 = affine.apply #map()[%1#0]
    %3 = affine.apply #map1()[%1#1]
    %padded = tensor.pad %0 low[0, 0] high[%2, %3] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %cst : f32
    } : tensor<1025x257xf32> to tensor<?x?xf32>
    %4 = iree_linalg_ext.set_encoding %padded : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS, original_type = tensor<1025x257xf32>>>
    %5:2 = iree_linalg_ext.upper_bound_tile_size tensor<257x513xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RHS>> -> index, index
    %6 = affine.apply #map1()[%5#0]
    %7 = affine.apply #map2()[%5#1]
    %padded_1 = tensor.pad %cst_0 low[0, 0] high[%6, %7] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %cst : f32
    } : tensor<257x513xf32> to tensor<?x?xf32>
    %8 = iree_linalg_ext.set_encoding %padded_1 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RHS, original_type = tensor<257x513xf32>>>
    %9:2 = iree_linalg_ext.upper_bound_tile_size tensor<1025x513xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT>> -> index, index
    %10 = affine.apply #map3()[%9#0]
    %11 = affine.apply #map4()[%9#1]
    %12 = tensor.empty(%10, %11) : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT, original_type = tensor<1025x513xf32>>>
    %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT, original_type = tensor<1025x513xf32>>>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT, original_type = tensor<1025x513xf32>>>
    %14 = linalg.matmul ins(%4, %8 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS, original_type = tensor<1025x257xf32>>>, tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RHS, original_type = tensor<257x513xf32>>>) outs(%13 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT, original_type = tensor<1025x513xf32>>>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT, original_type = tensor<1025x513xf32>>>
    %15 = iree_linalg_ext.unset_encoding %14 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT, original_type = tensor<1025x513xf32>>> -> tensor<?x?xf32>
    %extracted_slice = tensor.extract_slice %15[0, 0] [1025, 513] [1, 1] : tensor<?x?xf32> to tensor<1025x513xf32>
    %16 = hal.tensor.export %extracted_slice "output 0" : tensor<1025x513xf32> -> !hal.buffer_view
    return %16 : !hal.buffer_view
  }
}
// CHECK-LABEL: func.func @matmul_1025x513x257
// CHECK:         tensor.pack
// CHECK:         tensor.pack
// CHECK:         linalg.mmt4d
// CHECK:         tensor.unpack

// -----

#map = affine_map<()[s0] -> ((1025 ceildiv s0) * s0 - 1025)>
#map1 = affine_map<()[s0] -> ((257 ceildiv s0) * s0 - 257)>
#map2 = affine_map<()[s0] -> ((513 ceildiv s0) * s0 - 513)>
#map3 = affine_map<()[s0] -> ((1025 ceildiv s0) * s0)>
#map4 = affine_map<()[s0] -> ((513 ceildiv s0) * s0)>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader, GroupNonUniform], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, api=Vulkan, #spirv.resource_limits<max_compute_workgroup_size = [128, 128, 64], subgroup_size = 64, cooperative_matrix_properties_nv = []>>}>
#device_target_vulkan = #hal.device.target<"vulkan", {executable_targets = [#executable_target_vulkan_spirv_fb], legacy_sync}>
module attributes {hal.device.targets = [#device_target_vulkan]} {
  func.func @matmul_1025x513x257(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<257x513xf32>
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<1025x257xf32>
    %1:2 = iree_linalg_ext.upper_bound_tile_size tensor<1025x257xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS>> -> index, index
    %2 = affine.apply #map()[%1#0]
    %3 = affine.apply #map1()[%1#1]
    %padded = tensor.pad %0 low[0, 0] high[%2, %3] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %cst : f32
    } : tensor<1025x257xf32> to tensor<?x?xf32>
    %4 = iree_linalg_ext.set_encoding %padded : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS, original_type = tensor<1025x257xf32>>>
    %5:2 = iree_linalg_ext.upper_bound_tile_size tensor<257x513xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RHS>> -> index, index
    %6 = affine.apply #map1()[%5#0]
    %7 = affine.apply #map2()[%5#1]
    %padded_1 = tensor.pad %cst_0 low[0, 0] high[%6, %7] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %cst : f32
    } : tensor<257x513xf32> to tensor<?x?xf32>
    %8 = iree_linalg_ext.set_encoding %padded_1 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RHS, original_type = tensor<257x513xf32>>>
    %9:2 = iree_linalg_ext.upper_bound_tile_size tensor<1025x513xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT>> -> index, index
    %10 = affine.apply #map3()[%9#0]
    %11 = affine.apply #map4()[%9#1]
    %12 = tensor.empty(%10, %11) : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT, original_type = tensor<1025x513xf32>>>
    %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT, original_type = tensor<1025x513xf32>>>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT, original_type = tensor<1025x513xf32>>>
    %14 = linalg.matmul ins(%4, %8 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS, original_type = tensor<1025x257xf32>>>, tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RHS, original_type = tensor<257x513xf32>>>) outs(%13 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT, original_type = tensor<1025x513xf32>>>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT, original_type = tensor<1025x513xf32>>>
    %15 = iree_linalg_ext.unset_encoding %14 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT, original_type = tensor<1025x513xf32>>> -> tensor<?x?xf32>
    %extracted_slice = tensor.extract_slice %15[0, 0] [1025, 513] [1, 1] : tensor<?x?xf32> to tensor<1025x513xf32>
    %16 = hal.tensor.export %extracted_slice "output 0" : tensor<1025x513xf32> -> !hal.buffer_view
    return %16 : !hal.buffer_view
  }
}
// vulkan does not implement buildMaterializeEncodingsPassPipeline method.
// CHECK-LABEL: func.func @matmul_1025x513x257
// CHECK:         iree_linalg_ext.upper_bound_tile_size
// CHECK:         iree_linalg_ext.set_encoding
// CHECK:         iree_linalg_ext.upper_bound_tile_size
// CHECK:         iree_linalg_ext.set_encoding
// CHECK:         iree_linalg_ext.unset_encoding
