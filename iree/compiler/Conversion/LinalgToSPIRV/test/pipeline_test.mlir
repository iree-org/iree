// RUN: iree-opt -split-input-file -pass-pipeline="iree-codegen-linalg-to-spirv-pipeline" -iree-spirv-enable-vectorization %s | IreeFileCheck %s

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
    linalg.matmul ins(%arg0, %arg1 : memref<4096x4096xf32>, memref<4096x4096xf32>)
                 outs(%ret0 : memref<4096x4096xf32>)
    return
  }
  func @matmul_static_shape__num_workgroups__
    (!shapex.ranked_shape<[4096, 4096]>, !shapex.ranked_shape<[4096, 4096]>,
     !shapex.ranked_shape<[4096, 4096]>) -> (index, index, index)
    attributes {sym_visibility = "private"}
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

//    CHECK-LABEL: spv.func @matmul_static_shape
// CHECK-COUNT-8:    spv.Load "StorageBuffer" %{{.*}} : vector<4xf32>
//          CHECK:   spv.loop
// CHECK-COUNT-12:   spv.Load "StorageBuffer" %{{.*}} : vector<4xf32>
// CHECK-COUNT-32:   spv.FMul %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK-COUNT-8:   spv.Store "StorageBuffer" %{{.*}}, %{{.*}} : vector<4xf32>


