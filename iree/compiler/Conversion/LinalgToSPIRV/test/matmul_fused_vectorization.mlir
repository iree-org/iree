// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-spirv-linalg-tile-and-distribute,iree-spirv-tile-and-vectorize-in-one-workgroup,canonicalize,cse))" -iree-spirv-enable-vectorization %s | IreeFileCheck %s

hal.executable @matmul_static_shape attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @matmul_static_shape attributes {
      interface = @legacy_io, ordinal = 0 : index,
      signature = (!flow.dispatch.tensor<readonly:?x?xf32>, !flow.dispatch.tensor<readonly:?x?xf32>,
        !flow.dispatch.tensor<writeonly:?x?xf32>) -> ()}
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.3,
          [Shader, Float64, Float16, Int64, Int16, Int8, StorageBuffer16BitAccess,
           StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess,
           UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform,
           GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot,
           GroupNonUniformShuffle, GroupNonUniformShuffleRelative, VariablePointers,
           VariablePointersStorageBuffer],
          [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage,
           SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>,
          ARM:IntegratedGPU,
          {max_compute_shared_memory_size = 32768 : i32,
           max_compute_workgroup_invocations = 512 : i32,
           max_compute_workgroup_size = dense<512> : vector<3xi32>,
           subgroup_size = 16 : i32}>} {
      func @matmul_static_shape()
        attributes {vkspv.num_workgroups_fn = @matmul_static_shape__num_workgroups__} {
        %arg0 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@arg0, operand_result_num = 0 : i32} : memref<4096x4096xf32>
        %arg1 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@arg1, operand_result_num = 1 : i32} : memref<4096x4096xf32>
        %ret0 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@ret0, operand_result_num = 2 : i32} : memref<4096x4096xf32>
        %cst = constant 0.000000e+00 : f32
        linalg.fill(%ret0, %cst) : memref<4096x4096xf32>, f32
        linalg.matmul ins(%arg0, %arg1 : memref<4096x4096xf32>, memref<4096x4096xf32>)
                     outs(%ret0 : memref<4096x4096xf32>)
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
//    CHECK-LABEL: func @matmul_static_shape
//  CHECK-COUNT-8:   vector.transfer_write
//  CHECK-COUNT-8:   vector.transfer_read
//          CHECK:   %[[FOR_RES:.+]]:8 = scf.for
// CHECK-COUNT-12:     vector.transfer_read
// CHECK-COUNT-32:     vector.contract
//      CHECK:         scf.yield
//  CHECK-COUNT-8:    vector.transfer_write %[[FOR_RES]]
//          CHECK:    return
