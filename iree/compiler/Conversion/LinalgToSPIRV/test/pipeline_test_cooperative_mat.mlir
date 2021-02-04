// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-linalg-to-spirv-pipeline))" -iree-spirv-enable-vectorization %s | IreeFileCheck %s
// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-linalg-to-spirv-pipeline))" -iree-spirv-enable-vectorization -iree-spirv-use-workgroup-memory %s | IreeFileCheck %s

hal.executable @matmul_static_shape attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @matmul_static_shape attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x?xf32>, !flow.dispatch.input<?x?xf32>,
        !flow.dispatch.output<?x?xf32>) -> ()}
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.5,
          [Shader, Float64, Float16, Int64, Int16, Int8, StorageBuffer16BitAccess,
           StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess,
           UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform,
           GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot,
           GroupNonUniformShuffle, GroupNonUniformShuffleRelative, VariablePointers,
           VariablePointersStorageBuffer, CooperativeMatrixNV],
          [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage,
           SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers,
           SPV_NV_cooperative_matrix]>, NVIDIA:DiscreteGPU,
          {cooperative_matrix_properties_nv = [
            {a_type = i8, b_type = i8, c_type = i32, k_size = 32 : i32,
             m_size = 8 : i32, n_size = 8 : i32, result_type = i32, scope = 3 : i32},
            {a_type = f16, b_type = f16, c_type = f16, k_size = 16 : i32,
             m_size = 16 : i32, n_size = 16 : i32, result_type = f16,
             scope = 3 : i32},
            {a_type = f16, b_type = f16, c_type = f32, k_size = 16 : i32,
             m_size = 16 : i32, n_size = 16 : i32, result_type = f32,
             scope = 3 : i32}],
           max_compute_shared_memory_size = 49152 : i32,
           max_compute_workgroup_invocations = 1024 : i32,
           max_compute_workgroup_size = dense<[2147483647, 65535, 65535]> : vector<3xi32>,
           subgroup_size = 32 : i32}>} {
      func @matmul_static_shape()
        attributes {vkspv.num_workgroups_fn = @matmul_static_shape__num_workgroups__} {
        %arg0 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@arg0, operand_result_num = 0 : i32} : memref<4096x4096xf16>
        %arg1 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@arg1, operand_result_num = 1 : i32} : memref<4096x4096xf16>
        %ret0 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@ret0, operand_result_num = 2 : i32} : memref<4096x4096xf16>
        linalg.matmul ins(%arg0, %arg1 : memref<4096x4096xf16>, memref<4096x4096xf16>)
                     outs(%ret0 : memref<4096x4096xf16>)
        return
      }
      func private @matmul_static_shape__num_workgroups__
        (!shapex.ranked_shape<[4096, 4096]>, !shapex.ranked_shape<[4096, 4096]>,
         !shapex.ranked_shape<[4096, 4096]>) -> (index, index, index)
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//    CHECK-LABEL: spv.func @matmul_static_shape
// CHECK-COUNT-16:   spv.CooperativeMatrixLoadNV
//          CHECK:   spv.loop
// CHECK-COUNT-16:   spv.CooperativeMatrixLoadNV
// CHECK-COUNT-32:   spv.CooperativeMatrixMulAddNV
// CHECK-COUNT-16:   spv.CooperativeMatrixStoreNV
