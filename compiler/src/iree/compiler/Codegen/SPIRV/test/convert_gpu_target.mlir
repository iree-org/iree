// RUN: iree-opt --split-input-file --iree-gpu-test-target=valhall1 --pass-pipeline='builtin.module(iree-spirv-convert-gpu-target)' %s | FileCheck %s

// --iree-gpu-test-target is the flag for test cases that do not have
// `#iree_gpu.target` in IR. For test cases that have `#iree_gpu.target` in IR,
// the flag is ignored and the target info in IR is used directly.

#executable_target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
    iree_codegen.target_info = #iree_gpu.target<arch = "rdna3", features = "spirv:v1.6,cap:Shader",
      wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic, dot = dp4xi8toi32, mma = [<WMMAR3_F32_16x16x16_F16>, <WMMAR3_F16_16x16x16_F16>],
      subgroup_size_choices = [32, 64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
      max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>}>
func.func @dispatch() attributes { hal.executable.target = #executable_target } {
  return
}
//      CHECK: module attributes
// CHECK-SAME: spirv.target_env = #spirv.target_env<#spirv.vce<v1.6,
// CHECK-SAME:   [Shader, Float64, Float16, Int64, Int16, Int8,
// CHECK-SAME:    StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16,
// CHECK-SAME:    StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8,
// CHECK-SAME:    GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformArithmetic,
// CHECK-SAME:    DotProduct, DotProductInput4x8BitPacked, DotProductInputAll, DotProductInput4x8Bit,
// CHECK-SAME:    CooperativeMatrixKHR],
// CHECK-SAME:   [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_integer_dot_product, SPV_KHR_cooperative_matrix]>,
// CHECK-SAME:   AMD,
// CHECK-SAME:   #spirv.resource_limits<max_compute_shared_memory_size = 65536,
// CHECK-SAME:     max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024 : i32, 1024 : i32, 1024 : i32],
// CHECK-SAME:     min_subgroup_size = 32, max_subgroup_size = 64,
// CHECK-SAME:     cooperative_matrix_properties_khr = [
// CHECK-SAME:       #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f32, result_type = f32, acc_sat = false, scope = <Subgroup>>,
// CHECK-SAME:       #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f16, result_type = f16, acc_sat = false, scope = <Subgroup>>
// CHECK-SAME: ]>>


// -----

// Check that we filter out types not supported by VK_KHR_cooperative_matrix,
// e.g., bf16 and various versions of f8.

#executable_target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
    iree_codegen.target_info = #iree_gpu.target<arch = "rdna4", features = "spirv:v1.6,cap:Shader",
      wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b32, subgroup = none,
             mma = [<WMMAR4_F32_16x16x16_F16>, <WMMAR4_F16_16x16x16_F16>, <WMMAR4_BF16_16x16x16_BF16>, <WMMAR4_F32_16x16x16_BF16>,
                    <WMMAR4_F32_16x16x16_F8E5M2>, <WMMAR4_F32_16x16x16_F8E4M3FN_F8E5M2>, <WMMAR4_I32_16x16x16_I8>],
      subgroup_size_choices = [32, 64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
      max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>}>
func.func @dispatch_with_unsupported_types() attributes { hal.executable.target = #executable_target } {
  return
}
//      CHECK: module attributes
// CHECK-SAME: spirv.target_env = #spirv.target_env<#spirv.vce<v1.6,
// CHECK-SAME:   [Shader, Float64, Float16, Int64, Int16, Int8, CooperativeMatrixKHR],
// CHECK-SAME:   [SPV_KHR_cooperative_matrix]>,
// CHECK-SAME:   AMD,
// CHECK-SAME:   #spirv.resource_limits<max_compute_shared_memory_size = 65536,
// CHECK-SAME:     max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024 : i32, 1024 : i32, 1024 : i32],
// CHECK-SAME:     min_subgroup_size = 32, max_subgroup_size = 64,
// CHECK-SAME:     cooperative_matrix_properties_khr = [
// CHECK-SAME:       #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f32, result_type = f32, acc_sat = false, scope = <Subgroup>>,
// CHECK-SAME:       #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f16, result_type = f16, acc_sat = false, scope = <Subgroup>>,
// CHECK-SAME:       #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = i8, b_type = i8, c_type = i32, result_type = i32, acc_sat = false, scope = <Subgroup>>
// CHECK-SAME: ]>>

// -----

func.func @cli_target_without_executable_target() {
  return
}
//      CHECK: module attributes
// CHECK-SAME: spirv.target_env = #spirv.target_env<#spirv.vce<v1.6,

// -----
#executable_target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb">
func.func @cli_target_with_basic_executable_target() attributes { hal.executable.target = #executable_target } {
  return
}
//      CHECK: module attributes
// CHECK-SAME: spirv.target_env = #spirv.target_env<#spirv.vce<v1.6,
