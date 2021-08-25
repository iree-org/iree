// RUN: iree-opt -pass-pipeline='iree-hal-transformation-pipeline{serialize-executables=false}' -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s -check-prefix=DEFAULT
// RUN: iree-opt -pass-pipeline='iree-hal-transformation-pipeline{serialize-executables=false}' -iree-hal-target-backends=vulkan-spirv -iree-vulkan-target-triple=adreno-a650-android11 %s | IreeFileCheck %s -check-prefix=ADRENO
// RUN: iree-opt -pass-pipeline='iree-hal-transformation-pipeline{serialize-executables=false}' -iree-hal-target-backends=vulkan-spirv -iree-vulkan-target-triple=valhall-unknown-android11 %s | IreeFileCheck %s -check-prefix=MALI
// RUN: iree-opt -pass-pipeline='iree-hal-transformation-pipeline{serialize-executables=false}' -iree-hal-target-backends=vulkan-spirv -iree-vulkan-target-triple=turing-t4-linux %s | IreeFileCheck %s -check-prefix=TURINGT4
// RUN: iree-opt -pass-pipeline='iree-hal-transformation-pipeline{serialize-executables=false}' -iree-hal-target-backends=vulkan-spirv -iree-vulkan-target-triple=rdna1-5700xt-windows %s | IreeFileCheck %s -check-prefix=AMD5700XT

// TODO(antiagainst): Passing in lenghty strings as command-line options is not
// optimal. We should consider creating a dedicated test pass to pick up
// #vk.target_env in input assembly and convert them.

// DEFAULT: #spv.target_env<#spv.vce<v1.3, [Shader, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative], [SPV_KHR_storage_buffer_storage_class]>, SwiftShader:CPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 16384 : i32, max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>, subgroup_size = 4 : i32}>
// ADRENO: #spv.target_env<#spv.vce<v1.4, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, Qualcomm:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 1024 : i32, max_compute_workgroup_size = dense<[1024, 1024, 64]> : vector<3xi32>, subgroup_size = 64 : i32}>
// MALI: #spv.target_env<#spv.vce<v1.4, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformClustered, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>
// TURINGT4: #spv.target_env<#spv.vce<v1.5, [Shader, Float64, Float16, Int64, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformClustered, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer, CooperativeMatrixNV], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers, SPV_NV_cooperative_matrix]>, NVIDIA:DiscreteGPU, {cooperative_matrix_properties_nv = [{a_type = i8, b_type = i8, c_type = i32, k_size = 32 : i32, m_size = 8 : i32, n_size = 8 : i32, result_type = i32, scope = 3 : i32}, {a_type = f16, b_type = f16, c_type = f16, k_size = 16 : i32, m_size = 16 : i32, n_size = 16 : i32, result_type = f16, scope = 3 : i32}, {a_type = f16, b_type = f16, c_type = f32, k_size = 16 : i32, m_size = 16 : i32, n_size = 16 : i32, result_type = f32, scope = 3 : i32}], max_compute_shared_memory_size = 49152 : i32, max_compute_workgroup_invocations = 1024 : i32, max_compute_workgroup_size = dense<[1024, 1024, 64]> : vector<3xi32>, subgroup_size = 32 : i32}>
// AMD5700XT: #spv.target_env<#spv.vce<v1.5, [Shader, Float64, Float16, Int64, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformClustered, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, AMD:DiscreteGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 65536 : i32, max_compute_workgroup_invocations = 1024 : i32, max_compute_workgroup_size = dense<1024> : vector<3xi32>, subgroup_size = 64 : i32}>
#map0 = affine_map<(d0) -> (d0)>
flow.executable @simpleMath_dispatch_0 {
  flow.dispatch.entry @simpleMath_dispatch_0 attributes {workgroup_rank = 3 : index}
  module {
    func @simpleMath_dispatch_0(%arg0: !flow.dispatch.tensor<readonly:4xf32>, %arg1: !flow.dispatch.tensor<writeonly:4xf32>) {
      %c4 = constant 4 : index
      %c1 = constant 1 : index
      %c0 = constant 0 : index
      %0 = flow.dispatch.tensor.load %arg0, offsets = [%c0], sizes = [%c4], strides = [%c1] : !flow.dispatch.tensor<readonly:4xf32> -> tensor<4xf32>
      %1 = linalg.init_tensor [4] : tensor<4xf32>
      %2 = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%0 : tensor<4xf32>)
        outs(%1 : tensor<4xf32>)  {
      ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
        %3 = addf %arg3, %arg3 : f32
        linalg.yield %3 : f32
      } -> tensor<4xf32>
      flow.dispatch.tensor.store %2, %arg1, offsets = [%c0], sizes = [%c4], strides = [%c1] : tensor<4xf32> -> !flow.dispatch.tensor<writeonly:4xf32>
      return
    }
  }
}
