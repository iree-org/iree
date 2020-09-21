// RUN: iree-opt -pass-pipeline='iree-hal-transformation-pipeline{serialize-executables=false}' -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s -check-prefix=DEFAULT
// RUN: iree-opt -pass-pipeline='iree-hal-transformation-pipeline{serialize-executables=false}' -iree-hal-target-backends=vulkan-spirv -iree-vulkan-target-triple=qualcomm-adreno640-unknown-android10 %s | IreeFileCheck %s -check-prefix=ADRENO640
// RUN: iree-opt -pass-pipeline='iree-hal-transformation-pipeline{serialize-executables=false}' -iree-hal-target-backends=vulkan-spirv -iree-vulkan-target-triple=valhall-g77-unknown-android10 %s | IreeFileCheck %s -check-prefix=MALIG77

// TODO(antiagainst): Passing in lenghty strings as command-line options is not
// optimal. We should consider creating a dedicated test pass to pick up
// #vk.target_env in input assembly and convert them.

// DEFAULT: #spv.target_env<#spv.vce<v1.3, [Shader, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative], [SPV_KHR_storage_buffer_storage_class]>, SwiftShader:CPU, {max_compute_shared_memory_size = 16384 : i32, max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>, subgroup_size = 4 : i32}>
// ADRENO640: #spv.target_env<#spv.vce<v1.3, [Shader, Int16, GroupNonUniform, GroupNonUniformVote, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, Qualcomm:IntegratedGPU, {max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 1024 : i32, max_compute_workgroup_size = dense<[1024, 1024, 64]> : vector<3xi32>, subgroup_size = 64 : i32}>
// MALIG77: #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>
flow.executable @simpleMath_ex_dispatch_0 {
  flow.dispatch.entry @simpleMath_rgn_dispatch_0 attributes {
      workload = 4 : index
  }
  module {
    func @simpleMath_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
