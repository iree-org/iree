// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-linalg-to-spirv-pipeline))" -iree-spirv-enable-vectorization %s | IreeFileCheck %s

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
      func private @matmul_static_shape__num_workgroups__
        (!shapex.ranked_shape<[4096, 4096]>, !shapex.ranked_shape<[4096, 4096]>,
         !shapex.ranked_shape<[4096, 4096]>) -> (index, index, index)
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
      }
    }
  }
}
//    CHECK-LABEL: spv.func @matmul_static_shape
// CHECK-COUNT-8:    spv.Load "StorageBuffer" %{{.*}} : vector<4xf32>
//          CHECK:   spv.loop
// CHECK-COUNT-12:   spv.Load "StorageBuffer" %{{.*}} : vector<4xf32>
// CHECK-COUNT-32:   spv.FMul %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK-COUNT-8:   spv.Store "StorageBuffer" %{{.*}}, %{{.*}} : vector<4xf32>

// -----

hal.executable @matmul_fill_fused attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @matmul_fill_fused attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x?xf32>, !flow.dispatch.input<?x?xf32>,
        !flow.dispatch.output<?x?xf32>) -> ()}
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
      func @matmul_fill_fused()
        attributes {vkspv.num_workgroups_fn = @matmul_fill_fused__num_workgroups__} {
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
      func private @matmul_fill_fused__num_workgroups__
        (!shapex.ranked_shape<[4096, 4096]>, !shapex.ranked_shape<[4096, 4096]>,
         !shapex.ranked_shape<[4096, 4096]>) -> (index, index, index)
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
      }
    }
  }
}
//    CHECK-LABEL: spv.func @matmul_fill_fused
//      CHECK-NOT:   spv.Store "StorageBuffer"
//      CHECK-NOT:   spv.Load "StorageBuffer"
//          CHECK:   spv.loop
// CHECK-COUNT-12:   spv.Load "StorageBuffer" %{{.*}} : vector<4xf32>
// CHECK-COUNT-32:   spv.FMul %{{.*}}, %{{.*}} : vector<4xf32>
//  CHECK-COUNT-8:   spv.Store "StorageBuffer" %{{.*}}, %{{.*}} : vector<4xf32>

// -----

hal.executable @matmul_add_fused attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @matmul_add_fused attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x?xf32>, !flow.dispatch.input<?x?xf32>,
        !flow.dispatch.output<?x?xf32>) -> ()}
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
      func @matmul_add_fused() attributes {hal.num_workgroups_fn = @matmul_add_fused__num_workgroups__} {
        %cst = constant 0.000000e+00 : f32
        %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0, operand_result_index = 3 : i32} : memref<1024x256xf32>
        %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0, operand_result_index = 0 : i32} : memref<1024x512xf32>
        %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1, operand_result_index = 1 : i32} : memref<512x256xf32>
        %3 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg2, operand_result_index = 2 : i32} : memref<1024x256xf32>
        %4 = alloc() : memref<1024x256xf32>
        linalg.fill(%4, %cst) : memref<1024x256xf32>, f32
        linalg.matmul ins(%1, %2 : memref<1024x512xf32>, memref<512x256xf32>) 
        outs(%4 : memref<1024x256xf32>)
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
        affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]} 
        ins(%4, %3 : memref<1024x256xf32>, memref<1024x256xf32>) 
        outs(%0 : memref<1024x256xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
          %5 = addf %arg0, %arg1 : f32
          linalg.yield %5 : f32
        }
        return
      }
      func private @matmul_add_fused__num_workgroups__
        (!shapex.ranked_shape<[4096, 4096]>, !shapex.ranked_shape<[4096, 4096]>,
         !shapex.ranked_shape<[4096, 4096]>) -> (index, index, index)
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @arg2, set=0, binding=2, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//    CHECK-LABEL: spv.func @matmul_add_fused
//      CHECK-NOT:   spv.Store "StorageBuffer"
//      CHECK-NOT:   spv.Load "StorageBuffer"
//          CHECK:   spv.loop
// CHECK-COUNT-12:     spv.Load "StorageBuffer" %{{.*}} : vector<4xf32>
// CHECK-COUNT-32:     spv.FMul %{{.*}}, %{{.*}} : vector<4xf32>
//          CHECK:   spv.mlir.merge
//  CHECK-COUNT-8:   spv.Load "StorageBuffer" %{{.*}} : vector<4xf32>
//      CHECK-NOT:   spv.Load "StorageBuffer"
//      CHECK-NOT:   spv.Store "StorageBuffer"
//  CHECK-COUNT-8:   spv.FAdd %{{.*}}, %{{.*}} : vector<4xf32>
//  CHECK-COUNT-8:   spv.Store "StorageBuffer" %{{.*}}, %{{.*}} : vector<4xf32>
