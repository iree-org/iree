// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-transformation-pipeline{serialize-executables=false})' --iree-hal-target-backends=vulkan-spirv %s | FileCheck %s --check-prefix=DEFAULT
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-transformation-pipeline{serialize-executables=false})' --iree-hal-target-backends=vulkan-spirv --iree-vulkan-target-triple=adreno-a650-android30 %s | FileCheck %s --check-prefix=ADRENO
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-transformation-pipeline{serialize-executables=false})' --iree-hal-target-backends=vulkan-spirv --iree-vulkan-target-triple=valhall-unknown-android31 %s | FileCheck %s --check-prefix=VALHALL
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-transformation-pipeline{serialize-executables=false})' --iree-hal-target-backends=vulkan-spirv --iree-vulkan-target-triple=turing-t4-linux %s | FileCheck %s --check-prefix=TURING
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-transformation-pipeline{serialize-executables=false})' --iree-hal-target-backends=vulkan-spirv --iree-vulkan-target-triple=rdna1-5700xt-windows %s | FileCheck %s --check-prefix=RDNA1
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-transformation-pipeline{serialize-executables=false})' --iree-hal-target-backends=vulkan-spirv --iree-vulkan-target-triple=rdna3-6900xtx-windows %s | FileCheck %s --check-prefix=RDNA3
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-transformation-pipeline{serialize-executables=false})' --iree-hal-target-backends=vulkan-spirv --iree-vulkan-target-triple=m1-moltenvk-macos %s | FileCheck %s --check-prefix=M1
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-transformation-pipeline{serialize-executables=false})' --iree-hal-target-backends=vulkan-spirv --iree-vulkan-target-triple=arc-770-windows %s | FileCheck %s --check-prefix=ARC
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-transformation-pipeline{serialize-executables=false})' --iree-hal-target-backends=vulkan-spirv --iree-vulkan-target-triple=pascal-1080-windows %s | FileCheck %s --check-prefix=PASCAL

// TODO(antiagainst): Passing in lenghty strings as command-line options is not
// optimal. We should consider creating a dedicated test pass to pick up
// #vk.target_env in input assembly and convert them.

// DEFAULT: #spirv.target_env<#spirv.vce<v1.3,
// DEFAULT-SAME: [Shader, GroupNonUniform], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>,
// DEFAULT-SAME: api=Vulkan, #spirv.resource_limits<max_compute_workgroup_size = [128, 128, 64], subgroup_size = 64, cooperative_matrix_properties_nv = []>>

// ADRENO: #spirv.target_env<#spirv.vce<v1.4,
// ADRENO-SAME: [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer],
// ADRENO-SAME: [SPV_KHR_16bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>,
// ADRENO-SAME: api=Vulkan, Qualcomm:IntegratedGPU, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 64], subgroup_size = 64, cooperative_matrix_properties_nv = []>>

// VALHALL: #spirv.target_env<#spirv.vce<v1.4,
// VALHALL-SAME: [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformClustered, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer, DotProduct, DotProductInputAll, DotProductInput4x8BitPacked, DotProductInput4x8Bit],
// VALHALL-SAME: [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_integer_dot_product, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>,
// VALHALL-SAME: api=Vulkan, ARM:IntegratedGPU, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 512, max_compute_workgroup_size = [512, 512, 512], subgroup_size = 16, cooperative_matrix_properties_nv = []>>

// TURING: #spirv.target_env<#spirv.vce<v1.6,
// TURING-SAME: [Shader, Float64, Float16, Int64, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformClustered, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer, DotProduct, DotProductInputAll, DotProductInput4x8BitPacked, DotProductInput4x8Bit, CooperativeMatrixNV],
// TURING-SAME: [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_integer_dot_product, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers, SPV_NV_cooperative_matrix]>,
// TURING-SAME: api=Vulkan, NVIDIA:DiscreteGPU, #spirv.resource_limits<max_compute_shared_memory_size = 49152, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 64], min_subgroup_size = 32, max_subgroup_size = 32, cooperative_matrix_properties_nv = [#spirv.coop_matrix_props<m_size = 8, n_size = 8, k_size = 32, a_type = i8, b_type = i8, c_type = i32, result_type = i32, scope = <Subgroup>>, #spirv.coop_matrix_props<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f16, result_type = f16, scope = <Subgroup>>, #spirv.coop_matrix_props<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f32, result_type = f32, scope = <Subgroup>>]>>

// RDNA1: #spirv.target_env<#spirv.vce<v1.6,
// RDNA1-SAME: [Shader, Float64, Float16, Int64, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformClustered, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer, DotProduct, DotProductInputAll, DotProductInput4x8BitPacked, DotProductInput4x8Bit],
// RDNA1-SAME: [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_integer_dot_product, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>,
// RDNA1-SAME: api=Vulkan, AMD:DiscreteGPU, #spirv.resource_limits<max_compute_shared_memory_size = 65536, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 1024], subgroup_size = 64, min_subgroup_size = 32, max_subgroup_size = 64, cooperative_matrix_properties_nv = []>>

// RDNA3: #spirv.target_env<#spirv.vce<v1.6,
// RDNA3-SAME: [Shader, Float64, Float16, Int64, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformClustered, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer, DotProduct, DotProductInputAll, DotProductInput4x8BitPacked, DotProductInput4x8Bit, CooperativeMatrixNV],
// RDNA3-SAME: [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_integer_dot_product, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers, SPV_NV_cooperative_matrix]>,
// RDNA3-SAME: api=Vulkan, AMD:DiscreteGPU, #spirv.resource_limits<max_compute_shared_memory_size = 65536, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 1024], subgroup_size = 64, min_subgroup_size = 32, max_subgroup_size = 64, cooperative_matrix_properties_nv = [#spirv.coop_matrix_props<m_size = 16, n_size = 16, k_size = 16, a_type = i8, b_type = i8, c_type = i32, result_type = i32, scope = <Subgroup>>, #spirv.coop_matrix_props<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f16, result_type = f16, scope = <Subgroup>>, #spirv.coop_matrix_props<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f32, result_type = f32, scope = <Subgroup>>]>>

// M1: #spirv.target_env<#spirv.vce<v1.3,
// M1-SAME: [Shader, Float16, Int64, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer],
// M1-SAME: [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>,
// M1-SAME: api=Vulkan, Apple:IntegratedGPU, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 1024], cooperative_matrix_properties_nv = []>>

// ARC: #spirv.target_env<#spirv.vce<v1.6,
// ARC-SAME: [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformClustered, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer, DotProduct, DotProductInputAll, DotProductInput4x8BitPacked, DotProductInput4x8Bit],
// ARC-SAME: [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_integer_dot_product, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>,
// ARC-SAME: api=Vulkan, Intel:DiscreteGPU, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 64], min_subgroup_size = 8, max_subgroup_size = 32, cooperative_matrix_properties_nv = []>>}>

// PASCAL: #spirv.target_env<#spirv.vce<v1.6,
// PASCAL-SAME: [Shader, Float64, Float16, Int64, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformClustered, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer, DotProduct, DotProductInputAll, DotProductInput4x8BitPacked, DotProductInput4x8Bit],
// PASCAL-SAME: [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_integer_dot_product, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers, SPV_NV_cooperative_matrix]>,
// PASCAL-SAME: api=Vulkan, NVIDIA:DiscreteGPU, #spirv.resource_limits<max_compute_shared_memory_size = 49152, max_compute_workgroup_invocations = 1536, max_compute_workgroup_size = [1536, 1024, 64], min_subgroup_size = 32, max_subgroup_size = 32, cooperative_matrix_properties_nv = []>>}>

stream.executable public @reduce_dispatch {
  stream.executable.export @reduce_dispatch workgroups(%arg0: index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @reduce_dispatch(%arg0_binding: !stream.binding, %arg1_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %arg0 = stream.binding.subspan %arg0_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<16xf32>>
      %arg1 = stream.binding.subspan %arg1_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<f32>>
      %0 = tensor.empty() : tensor<f32>
      %1 = flow.dispatch.tensor.load %arg0, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
      %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%1 : tensor<16xf32>) outs(%0 : tensor<f32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %4 = arith.addf %arg2, %arg3 : f32
        linalg.yield %4 : f32
      } -> tensor<f32>
      flow.dispatch.tensor.store %3, %arg1, offsets=[], sizes=[], strides=[] : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>
      return
    }
  }
}
