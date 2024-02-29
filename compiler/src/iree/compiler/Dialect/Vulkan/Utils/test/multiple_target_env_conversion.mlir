// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetBackends=vulkan-spirv},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target-triple=rdna3-7900xtx-windows \
// RUN:   --iree-vulkan-target-env="#vk.target_env<v1.1, r(120), [VK_KHR_spirv_1_4, VK_KHR_storage_buffer_storage_class], AMD:DiscreteGPU, #vk.caps<maxComputeSharedMemorySize = 16384, maxComputeWorkGroupInvocations = 1024, maxComputeWorkGroupSize = dense<[128, 8, 4]>: vector<3xi32>, subgroupFeatures = 63 : i32, subgroupSize = 4 >>" \
// RUN:   --iree-vulkan-target-triple=valhall-unknown-android31 \
// RUN:   %s | FileCheck %s

// CHECK: #[[RDNA3:.+]] = {{.*}} #spirv.target_env<#spirv.vce<v1.6, [Shader, Float64, Float16, Int64, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformClustered, GroupNonUniformQuad, PhysicalStorageBufferAddresses, VariablePointers, VariablePointersStorageBuffer, DotProduct, DotProductInputAll, DotProductInput4x8BitPacked, DotProductInput4x8Bit, CooperativeMatrixKHR],
// CHECK-SAME: [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_integer_dot_product, SPV_KHR_storage_buffer_storage_class, SPV_KHR_physical_storage_buffer, SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>,
// CHECK-SAME: api=Vulkan, AMD:DiscreteGPU, #spirv.resource_limits<max_compute_shared_memory_size = 65536, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 1024], subgroup_size = 64, min_subgroup_size = 32, max_subgroup_size = 64, cooperative_matrix_properties_khr = [#spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = i8, b_type = i8, c_type = i32, result_type = i32, acc_sat = false, scope = <Subgroup>>, #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f16, result_type = f16, acc_sat = false, scope = <Subgroup>>, #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f32, result_type = f32, acc_sat = false, scope = <Subgroup>>]>>

// CHECK: #[[ENV:.+]] = {{.*}} #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative],
// CHECK-SAME: [SPV_KHR_storage_buffer_storage_class]>,
// CHECK-SAME: api=Vulkan, AMD:DiscreteGPU, #spirv.resource_limits<max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [128, 8, 4], subgroup_size = 4, cooperative_matrix_properties_khr = []>>

// CHECK: #[[VALHALL:.+]] = {{.*}} #spirv.target_env<#spirv.vce<v1.4, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformClustered, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer, DotProduct, DotProductInputAll, DotProductInput4x8BitPacked, DotProductInput4x8Bit],
// CHECK-SAME: [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_integer_dot_product, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>,
// CHECK-SAME: api=Vulkan, ARM:IntegratedGPU, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 512, max_compute_workgroup_size = [512, 512, 512], subgroup_size = 16, cooperative_matrix_properties_khr = []>>

// Verify that the order of target environments matches what the user specified.

// CHECK: [#[[RDNA3]], #[[ENV]], #[[VALHALL]]]

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
