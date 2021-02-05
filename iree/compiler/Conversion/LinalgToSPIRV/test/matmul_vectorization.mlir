// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-linalg-tile-and-fuse,canonicalize,cse))" -iree-spirv-enable-vectorization %s | IreeFileCheck %s
// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-linalg-tile-and-fuse,canonicalize,cse))" -iree-spirv-enable-vectorization -iree-spirv-use-workgroup-memory %s | IreeFileCheck %s -check-prefix=PROMOTE

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
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 64)>
//      CHECK: func @matmul_static_shape
//  CHECK-DAG:  %[[ARG0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg0
//  CHECK-DAG:  %[[ARG1:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg1
//  CHECK-DAG:  %[[RET0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@ret0
//  CHECK-DAG:  %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:  %[[CST:.+]] = constant 0.0
//  CHECK-DAG:  %[[C16:.+]] = constant 16 : index
//  CHECK-DAG:  %[[C32:.+]] = constant 32 : index
//  CHECK-DAG:  %[[C48:.+]] = constant 48 : index
//      CHECK:  %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//      CHECK:  %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//      CHECK:  %[[BOFFSET_Y:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//      CHECK:  %[[BOFFSET_X:.+]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//      CHECK:    %[[SUBVIEW_RESULT:.+]] = subview %[[RET0]]
// CHECK-SAME:      [%[[BOFFSET_Y]], %[[BOFFSET_X]]] [64, 64]
//      CHECK:  %[[SUBVIEW_RESULT_2:.+]] = subview %[[SUBVIEW_RESULT]]
// CHECK-SAME:    [0, 0] [64, 64] [1, 1]

//  CHECK-DAG:  %[[READ_INIT_0_0:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C0]]]
//  CHECK-DAG:  %[[READ_INIT_0_1:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C16]]]
//  CHECK-DAG:  %[[READ_INIT_0_2:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C32]]]
//  CHECK-DAG:  %[[READ_INIT_0_3:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C48]]]

//  CHECK-DAG:  %[[READ_INIT_1_0:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C16]], %[[C0]]]
//  CHECK-DAG:  %[[READ_INIT_1_1:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C16]], %[[C16]]]
//  CHECK-DAG:  %[[READ_INIT_1_2:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C16]], %[[C32]]]
//  CHECK-DAG:  %[[READ_INIT_1_3:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C16]], %[[C48]]]

//  CHECK-DAG:  %[[READ_INIT_2_0:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C32]], %[[C0]]]
//  CHECK-DAG:  %[[READ_INIT_2_1:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C32]], %[[C16]]]
//  CHECK-DAG:  %[[READ_INIT_2_2:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C32]], %[[C32]]]
//  CHECK-DAG:  %[[READ_INIT_2_3:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C32]], %[[C48]]]

//  CHECK-DAG:  %[[READ_INIT_3_0:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C48]], %[[C0]]]
//  CHECK-DAG:  %[[READ_INIT_3_1:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C48]], %[[C16]]]
//  CHECK-DAG:  %[[READ_INIT_3_2:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C48]], %[[C32]]]
//  CHECK-DAG:  %[[READ_INIT_3_3:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C48]], %[[C48]]]

//      CHECK:  %[[FOR_RES:.+]]:16 = scf.for %[[IV0:.+]] = {{.*}} to
// CHECK-SAME:  iter_args(%[[ACC_0_0:.+]] = %[[READ_INIT_0_0]],
// CHECK-SAME:  %[[ACC_0_1:.+]] = %[[READ_INIT_0_1]],
// CHECK-SAME:  %[[ACC_0_2:.+]] = %[[READ_INIT_0_2]],
// CHECK-SAME:  %[[ACC_0_3:.+]] = %[[READ_INIT_0_3]],
// CHECK-SAME:  %[[ACC_1_0:.+]] = %[[READ_INIT_1_0]],
// CHECK-SAME:  %[[ACC_1_1:.+]] = %[[READ_INIT_1_1]],
// CHECK-SAME:  %[[ACC_1_2:.+]] = %[[READ_INIT_1_2]],
// CHECK-SAME:  %[[ACC_1_3:.+]] = %[[READ_INIT_1_3]],
// CHECK-SAME:  %[[ACC_2_0:.+]] = %[[READ_INIT_2_0]],
// CHECK-SAME:  %[[ACC_2_1:.+]] = %[[READ_INIT_2_1]],
// CHECK-SAME:  %[[ACC_2_2:.+]] = %[[READ_INIT_2_2]],
// CHECK-SAME:  %[[ACC_2_3:.+]] = %[[READ_INIT_2_3]],
// CHECK-SAME:  %[[ACC_3_0:.+]] = %[[READ_INIT_3_0]],
// CHECK-SAME:  %[[ACC_3_1:.+]] = %[[READ_INIT_3_1]],
// CHECK-SAME:  %[[ACC_3_2:.+]] = %[[READ_INIT_3_2]],
// CHECK-SAME:  %[[ACC_3_3:.+]] = %[[READ_INIT_3_3]])
//      CHECK:    %[[SUBVIEW_LHS:.+]] = subview %[[ARG0]]
// CHECK-SAME:      [%[[BOFFSET_Y]], %[[IV0]]] [64, 32]
//      CHECK:    %[[SUBVIEW_RHS:.+]] = subview %[[ARG1]]
// CHECK-SAME:      [%[[IV0]], %[[BOFFSET_X]]] [32, 64]
//      CHECK:    %[[SUBVIEW_LHS_2:.+]] = subview %[[SUBVIEW_LHS]]
// CHECK-SAME:      [0, 0] [64, 32] [1, 1]
//      CHECK:    %[[SUBVIEW_RHS_2:.+]] = subview %[[SUBVIEW_RHS]]
// CHECK-SAME:      [0, 0] [32, 64] [1, 1]

//  CHECK-DAG:    %[[READ_LHS_0_0:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_LHS_2]][%[[C0]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_0_1:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_LHS_2]][%[[C0]], %[[C16]]]

//  CHECK-DAG:    %[[READ_LHS_1_0:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_LHS_2]][%[[C16]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_1_1:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_LHS_2]][%[[C16]], %[[C16]]]

//  CHECK-DAG:    %[[READ_LHS_2_0:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_LHS_2]][%[[C32]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_2_1:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_LHS_2]][%[[C32]], %[[C16]]]

//  CHECK-DAG:    %[[READ_LHS_3_0:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_LHS_2]][%[[C48]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_3_1:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_LHS_2]][%[[C48]], %[[C16]]]

//  CHECK-DAG:    %[[READ_RHS_0_0:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_RHS_2]][%[[C0]], %[[C0]]]
//  CHECK-DAG:    %[[READ_RHS_0_1:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_RHS_2]][%[[C0]], %[[C16]]]
//  CHECK-DAG:    %[[READ_RHS_0_2:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_RHS_2]][%[[C0]], %[[C32]]]
//  CHECK-DAG:    %[[READ_RHS_0_3:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_RHS_2]][%[[C0]], %[[C48]]]

//  CHECK-DAG:    %[[READ_RHS_1_0:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_RHS_2]][%[[C16]], %[[C0]]]
//  CHECK-DAG:    %[[READ_RHS_1_1:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_RHS_2]][%[[C16]], %[[C16]]]
//  CHECK-DAG:    %[[READ_RHS_1_2:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_RHS_2]][%[[C16]], %[[C32]]]
//  CHECK-DAG:    %[[READ_RHS_1_3:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_RHS_2]][%[[C16]], %[[C48]]]

//      CHECK:    %[[CONTRACT_0_0_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_0_0]], %[[READ_RHS_0_0]], %[[ACC_0_0]]
//      CHECK:    %[[CONTRACT_0_0:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_0_1]], %[[READ_RHS_1_0]], %[[CONTRACT_0_0_1]]
//      CHECK:    %[[CONTRACT_0_1_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_0_0]], %[[READ_RHS_0_1]], %[[ACC_0_1]]
//      CHECK:    %[[CONTRACT_0_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_0_1]], %[[READ_RHS_1_1]], %[[CONTRACT_0_1_1]]
//      CHECK:    %[[CONTRACT_0_2_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_0_0]], %[[READ_RHS_0_2]], %[[ACC_0_2]]
//      CHECK:    %[[CONTRACT_0_2:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_0_1]], %[[READ_RHS_1_2]], %[[CONTRACT_0_2_1]]
//      CHECK:    %[[CONTRACT_0_3_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_0_0]], %[[READ_RHS_0_3]], %[[ACC_0_3]]
//      CHECK:    %[[CONTRACT_0_3:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_0_1]], %[[READ_RHS_1_3]], %[[CONTRACT_0_3_1]]

//      CHECK:    %[[CONTRACT_1_0_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_1_0]], %[[READ_RHS_0_0]], %[[ACC_1_0]]
//      CHECK:    %[[CONTRACT_1_0:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_1_1]], %[[READ_RHS_1_0]], %[[CONTRACT_1_0_1]]
//      CHECK:    %[[CONTRACT_1_1_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_1_0]], %[[READ_RHS_0_1]], %[[ACC_1_1]]
//      CHECK:    %[[CONTRACT_1_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_1_1]], %[[READ_RHS_1_1]], %[[CONTRACT_1_1_1]]
//      CHECK:    %[[CONTRACT_1_2_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_1_0]], %[[READ_RHS_0_2]], %[[ACC_1_2]]
//      CHECK:    %[[CONTRACT_1_2:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_1_1]], %[[READ_RHS_1_2]], %[[CONTRACT_1_2_1]]
//      CHECK:    %[[CONTRACT_1_3_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_1_0]], %[[READ_RHS_0_3]], %[[ACC_1_3]]
//      CHECK:    %[[CONTRACT_1_3:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_1_1]], %[[READ_RHS_1_3]], %[[CONTRACT_1_3_1]]

//      CHECK:    %[[CONTRACT_2_0_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_2_0]], %[[READ_RHS_0_0]], %[[ACC_2_0]]
//      CHECK:    %[[CONTRACT_2_0:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_2_1]], %[[READ_RHS_1_0]], %[[CONTRACT_2_0_1]]
//      CHECK:    %[[CONTRACT_2_1_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_2_0]], %[[READ_RHS_0_1]], %[[ACC_2_1]]
//      CHECK:    %[[CONTRACT_2_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_2_1]], %[[READ_RHS_1_1]], %[[CONTRACT_2_1_1]]
//      CHECK:    %[[CONTRACT_2_2_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_2_0]], %[[READ_RHS_0_2]], %[[ACC_2_2]]
//      CHECK:    %[[CONTRACT_2_2:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_2_1]], %[[READ_RHS_1_2]], %[[CONTRACT_2_2_1]]
//      CHECK:    %[[CONTRACT_2_3_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_2_0]], %[[READ_RHS_0_3]], %[[ACC_2_3]]
//      CHECK:    %[[CONTRACT_2_3:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_2_1]], %[[READ_RHS_1_3]], %[[CONTRACT_2_3_1]]

//      CHECK:    %[[CONTRACT_3_0_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_3_0]], %[[READ_RHS_0_0]], %[[ACC_3_0]]
//      CHECK:    %[[CONTRACT_3_0:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_3_1]], %[[READ_RHS_1_0]], %[[CONTRACT_3_0_1]]
//      CHECK:    %[[CONTRACT_3_1_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_3_0]], %[[READ_RHS_0_1]], %[[ACC_3_1]]
//      CHECK:    %[[CONTRACT_3_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_3_1]], %[[READ_RHS_1_1]], %[[CONTRACT_3_1_1]]
//      CHECK:    %[[CONTRACT_3_2_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_3_0]], %[[READ_RHS_0_2]], %[[ACC_3_2]]
//      CHECK:    %[[CONTRACT_3_2:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_3_1]], %[[READ_RHS_1_2]], %[[CONTRACT_3_2_1]]
//      CHECK:    %[[CONTRACT_3_3_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_3_0]], %[[READ_RHS_0_3]], %[[ACC_3_3]]
//      CHECK:    %[[CONTRACT_3_3:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_3_1]], %[[READ_RHS_1_3]], %[[CONTRACT_3_3_1]]

//      CHECK:    scf.yield %[[CONTRACT_0_0]], %[[CONTRACT_0_1]],
// CHECK-SAME:      %[[CONTRACT_0_2]], %[[CONTRACT_0_3]], %[[CONTRACT_1_0]],
// CHECK-SAME:      %[[CONTRACT_1_1]], %[[CONTRACT_1_2]], %[[CONTRACT_1_3]],
// CHECK-SAME:      %[[CONTRACT_2_0]], %[[CONTRACT_2_1]], %[[CONTRACT_2_2]],
// CHECK-SAME:      %[[CONTRACT_2_3]], %[[CONTRACT_3_0]], %[[CONTRACT_3_1]],
// CHECK-SAME:      %[[CONTRACT_3_2]], %[[CONTRACT_3_3]]

//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#0, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#1, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C16]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#2, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C32]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#3, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C48]]]

//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#4, %[[SUBVIEW_RESULT_2]][%[[C16]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#5, %[[SUBVIEW_RESULT_2]][%[[C16]], %[[C16]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#6, %[[SUBVIEW_RESULT_2]][%[[C16]], %[[C32]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#7, %[[SUBVIEW_RESULT_2]][%[[C16]], %[[C48]]]

//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#8, %[[SUBVIEW_RESULT_2]][%[[C32]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#9, %[[SUBVIEW_RESULT_2]][%[[C32]], %[[C16]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#10, %[[SUBVIEW_RESULT_2]][%[[C32]], %[[C32]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#11, %[[SUBVIEW_RESULT_2]][%[[C32]], %[[C48]]]

//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#12, %[[SUBVIEW_RESULT_2]][%[[C48]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#13, %[[SUBVIEW_RESULT_2]][%[[C48]], %[[C16]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#14, %[[SUBVIEW_RESULT_2]][%[[C48]], %[[C32]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#15, %[[SUBVIEW_RESULT_2]][%[[C48]], %[[C48]]]


//  PROMOTE-DAG: #[[MAP4:.+]] = affine_map<()[s0] -> (s0 * 64 - (s0 floordiv 2) * 128)>
//      PROMOTE: func @matmul_static_shape
//  PROMOTE-DAG:  %[[ARG0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg0
//  PROMOTE-DAG:  %[[ARG1:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg1
//  PROMOTE-DAG:  %[[RET0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@ret0
//  PROMOTE-DAG:  %[[C0:.+]] = constant 0 : index
//  PROMOTE-DAG:  %[[C2:.+]] = constant 2
//  PROMOTE-DAG:  %[[C16:.+]] = constant 16
//  PROMOTE-DAG:  %[[C32:.+]] = constant 32
//  PROMOTE-DAG:  %[[C48:.+]] = constant 48
//  PROMOTE-DAG:  %[[ALLOC1:.+]] = alloc() : memref<128x32xf16, 3>
//  PROMOTE-DAG:  %[[ALLOC2:.+]] = alloc() : memref<32x128xf16, 3>

//      PROMOTE:  %[[RESULT_SUBVIEW:.+]] = subview %[[RET0]]
//      PROMOTE:  %[[WGMEM_LHS_SUBVIEW:.+]] = subview %[[ALLOC1]][0, 0] [128, 32] [1, 1]
//      PROMOTE:  %[[WGMEM_RHS_SUBVIEW:.+]] = subview %[[ALLOC2]][0, 0] [32, 128] [1, 1]
//      PROMOTE:  %[[SG_X:.+]] = gpu.subgroup_id
//      PROMOTE:  %[[SG_Y:.+]] = divi_signed %[[SG_X]], %[[C2]]
//      PROMOTE:  %[[SGOFFSET_Y:.+]] = affine.apply #[[MAP4]]()[%[[SG_Y]]]
//      PROMOTE:  %[[SG_LHS_SUBVIEW:.+]] = subview %[[WGMEM_LHS_SUBVIEW]][%[[SGOFFSET_Y]], 0]
//      PROMOTE:  %[[SGOFFSET_X:.+]] = affine.apply #[[MAP4]]()[%[[SG_X]]]
//      PROMOTE:  %[[SG_RHS_SUBVIEW:.+]] = subview %[[WGMEM_RHS_SUBVIEW]][0, %[[SGOFFSET_X]]]
//      PROMOTE:  %[[SG_RESULT_SUBVIEW:.+]] = subview %[[RESULT_SUBVIEW]][%[[SGOFFSET_Y]], %[[SGOFFSET_X]]]

//  PROMOTE-DAG:  %[[READ_INIT_0_0:.+]] = vector.transfer_read
// PROMOTE-SAME:    %[[SG_RESULT_SUBVIEW]][%[[C0]], %[[C0]]]
//  PROMOTE-DAG:  %[[READ_INIT_0_1:.+]] = vector.transfer_read
// PROMOTE-SAME:    %[[SG_RESULT_SUBVIEW]][%[[C0]], %[[C16]]]
//  PROMOTE-DAG:  %[[READ_INIT_0_2:.+]] = vector.transfer_read
// PROMOTE-SAME:    %[[SG_RESULT_SUBVIEW]][%[[C0]], %[[C32]]]
//  PROMOTE-DAG:  %[[READ_INIT_0_3:.+]] = vector.transfer_read
// PROMOTE-SAME:    %[[SG_RESULT_SUBVIEW]][%[[C0]], %[[C48]]]

//  PROMOTE-DAG:  %[[READ_INIT_1_0:.+]] = vector.transfer_read
// PROMOTE-SAME:    %[[SG_RESULT_SUBVIEW]][%[[C16]], %[[C0]]]
//  PROMOTE-DAG:  %[[READ_INIT_1_1:.+]] = vector.transfer_read
// PROMOTE-SAME:    %[[SG_RESULT_SUBVIEW]][%[[C16]], %[[C16]]]
//  PROMOTE-DAG:  %[[READ_INIT_1_2:.+]] = vector.transfer_read
// PROMOTE-SAME:    %[[SG_RESULT_SUBVIEW]][%[[C16]], %[[C32]]]
//  PROMOTE-DAG:  %[[READ_INIT_1_3:.+]] = vector.transfer_read
// PROMOTE-SAME:    %[[SG_RESULT_SUBVIEW]][%[[C16]], %[[C48]]]

//  PROMOTE-DAG:  %[[READ_INIT_2_0:.+]] = vector.transfer_read
// PROMOTE-SAME:    %[[SG_RESULT_SUBVIEW]][%[[C32]], %[[C0]]]
//  PROMOTE-DAG:  %[[READ_INIT_2_1:.+]] = vector.transfer_read
// PROMOTE-SAME:    %[[SG_RESULT_SUBVIEW]][%[[C32]], %[[C16]]]
//  PROMOTE-DAG:  %[[READ_INIT_2_2:.+]] = vector.transfer_read
// PROMOTE-SAME:    %[[SG_RESULT_SUBVIEW]][%[[C32]], %[[C32]]]
//  PROMOTE-DAG:  %[[READ_INIT_2_3:.+]] = vector.transfer_read
// PROMOTE-SAME:    %[[SG_RESULT_SUBVIEW]][%[[C32]], %[[C48]]]

//  PROMOTE-DAG:  %[[READ_INIT_3_0:.+]] = vector.transfer_read
// PROMOTE-SAME:    %[[SG_RESULT_SUBVIEW]][%[[C48]], %[[C0]]]
//  PROMOTE-DAG:  %[[READ_INIT_3_1:.+]] = vector.transfer_read
// PROMOTE-SAME:    %[[SG_RESULT_SUBVIEW]][%[[C48]], %[[C16]]]
//  PROMOTE-DAG:  %[[READ_INIT_3_2:.+]] = vector.transfer_read
// PROMOTE-SAME:    %[[SG_RESULT_SUBVIEW]][%[[C48]], %[[C32]]]
//  PROMOTE-DAG:  %[[READ_INIT_3_3:.+]] = vector.transfer_read
// PROMOTE-SAME:    %[[SG_RESULT_SUBVIEW]][%[[C48]], %[[C48]]]

//      PROMOTE:  %[[FOR_RES:.+]]:16 = scf.for %[[IV0:.+]] = {{.*}} to
// PROMOTE-SAME:  iter_args(%[[ACC_0_0:.+]] = %[[READ_INIT_0_0]],
// PROMOTE-SAME:  %[[ACC_0_1:.+]] = %[[READ_INIT_0_1]],
// PROMOTE-SAME:  %[[ACC_0_2:.+]] = %[[READ_INIT_0_2]],
// PROMOTE-SAME:  %[[ACC_0_3:.+]] = %[[READ_INIT_0_3]],
// PROMOTE-SAME:  %[[ACC_1_0:.+]] = %[[READ_INIT_1_0]],
// PROMOTE-SAME:  %[[ACC_1_1:.+]] = %[[READ_INIT_1_1]],
// PROMOTE-SAME:  %[[ACC_1_2:.+]] = %[[READ_INIT_1_2]],
// PROMOTE-SAME:  %[[ACC_1_3:.+]] = %[[READ_INIT_1_3]],
// PROMOTE-SAME:  %[[ACC_2_0:.+]] = %[[READ_INIT_2_0]],
// PROMOTE-SAME:  %[[ACC_2_1:.+]] = %[[READ_INIT_2_1]],
// PROMOTE-SAME:  %[[ACC_2_2:.+]] = %[[READ_INIT_2_2]],
// PROMOTE-SAME:  %[[ACC_2_3:.+]] = %[[READ_INIT_2_3]],
// PROMOTE-SAME:  %[[ACC_3_0:.+]] = %[[READ_INIT_3_0]],
// PROMOTE-SAME:  %[[ACC_3_1:.+]] = %[[READ_INIT_3_1]],
// PROMOTE-SAME:  %[[ACC_3_2:.+]] = %[[READ_INIT_3_2]],
// PROMOTE-SAME:  %[[ACC_3_3:.+]] = %[[READ_INIT_3_3]])

//      PROMOTE:    %[[LHS_SUBVIEW:.+]] = subview %[[ARG0]]
//      PROMOTE:    %[[RHS_SUBVIEW:.+]] = subview %[[ARG1]]
//      PROMOTE:    linalg.copy(%[[LHS_SUBVIEW]], %[[WGMEM_LHS_SUBVIEW]])
//      PROMOTE:    linalg.copy(%[[RHS_SUBVIEW]], %[[WGMEM_RHS_SUBVIEW]])

//  PROMOTE-DAG:    %[[READ_LHS_0_0:.+]] = vector.transfer_read
// PROMOTE-SAME:      %[[SG_LHS_SUBVIEW]][%[[C0]], %[[C0]]]
//  PROMOTE-DAG:    %[[READ_LHS_0_1:.+]] = vector.transfer_read
// PROMOTE-SAME:      %[[SG_LHS_SUBVIEW]][%[[C0]], %[[C16]]]

//  PROMOTE-DAG:    %[[READ_LHS_1_0:.+]] = vector.transfer_read
// PROMOTE-SAME:      %[[SG_LHS_SUBVIEW]][%[[C16]], %[[C0]]]
//  PROMOTE-DAG:    %[[READ_LHS_1_1:.+]] = vector.transfer_read
// PROMOTE-SAME:      %[[SG_LHS_SUBVIEW]][%[[C16]], %[[C16]]]

//  PROMOTE-DAG:    %[[READ_LHS_2_0:.+]] = vector.transfer_read
// PROMOTE-SAME:      %[[SG_LHS_SUBVIEW]][%[[C32]], %[[C0]]]
//  PROMOTE-DAG:    %[[READ_LHS_2_1:.+]] = vector.transfer_read
// PROMOTE-SAME:      %[[SG_LHS_SUBVIEW]][%[[C32]], %[[C16]]]

//  PROMOTE-DAG:    %[[READ_LHS_3_0:.+]] = vector.transfer_read
// PROMOTE-SAME:      %[[SG_LHS_SUBVIEW]][%[[C48]], %[[C0]]]
//  PROMOTE-DAG:    %[[READ_LHS_3_1:.+]] = vector.transfer_read
// PROMOTE-SAME:      %[[SG_LHS_SUBVIEW]][%[[C48]], %[[C16]]]

//  PROMOTE-DAG:    %[[READ_RHS_0_0:.+]] = vector.transfer_read
// PROMOTE-SAME:      %[[SG_RHS_SUBVIEW]][%[[C0]], %[[C0]]]
//  PROMOTE-DAG:    %[[READ_RHS_0_1:.+]] = vector.transfer_read
// PROMOTE-SAME:      %[[SG_RHS_SUBVIEW]][%[[C0]], %[[C16]]]
//  PROMOTE-DAG:    %[[READ_RHS_0_2:.+]] = vector.transfer_read
// PROMOTE-SAME:      %[[SG_RHS_SUBVIEW]][%[[C0]], %[[C32]]]
//  PROMOTE-DAG:    %[[READ_RHS_0_3:.+]] = vector.transfer_read
// PROMOTE-SAME:      %[[SG_RHS_SUBVIEW]][%[[C0]], %[[C48]]]

//  PROMOTE-DAG:    %[[READ_RHS_1_0:.+]] = vector.transfer_read
// PROMOTE-SAME:      %[[SG_RHS_SUBVIEW]][%[[C16]], %[[C0]]]
//  PROMOTE-DAG:    %[[READ_RHS_1_1:.+]] = vector.transfer_read
// PROMOTE-SAME:      %[[SG_RHS_SUBVIEW]][%[[C16]], %[[C16]]]
//  PROMOTE-DAG:    %[[READ_RHS_1_2:.+]] = vector.transfer_read
// PROMOTE-SAME:      %[[SG_RHS_SUBVIEW]][%[[C16]], %[[C32]]]
//  PROMOTE-DAG:    %[[READ_RHS_1_3:.+]] = vector.transfer_read
// PROMOTE-SAME:      %[[SG_RHS_SUBVIEW]][%[[C16]], %[[C48]]]

//      PROMOTE:    %[[CONTRACT_0_0_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_0_0]], %[[READ_RHS_0_0]], %[[ACC_0_0]]
//      PROMOTE:    %[[CONTRACT_0_0:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_0_1]], %[[READ_RHS_1_0]], %[[CONTRACT_0_0_1]]
//      PROMOTE:    %[[CONTRACT_0_1_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_0_0]], %[[READ_RHS_0_1]], %[[ACC_0_1]]
//      PROMOTE:    %[[CONTRACT_0_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_0_1]], %[[READ_RHS_1_1]], %[[CONTRACT_0_1_1]]
//      PROMOTE:    %[[CONTRACT_0_2_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_0_0]], %[[READ_RHS_0_2]], %[[ACC_0_2]]
//      PROMOTE:    %[[CONTRACT_0_2:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_0_1]], %[[READ_RHS_1_2]], %[[CONTRACT_0_2_1]]
//      PROMOTE:    %[[CONTRACT_0_3_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_0_0]], %[[READ_RHS_0_3]], %[[ACC_0_3]]
//      PROMOTE:    %[[CONTRACT_0_3:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_0_1]], %[[READ_RHS_1_3]], %[[CONTRACT_0_3_1]]

//      PROMOTE:    %[[CONTRACT_1_0_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_1_0]], %[[READ_RHS_0_0]], %[[ACC_1_0]]
//      PROMOTE:    %[[CONTRACT_1_0:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_1_1]], %[[READ_RHS_1_0]], %[[CONTRACT_1_0_1]]
//      PROMOTE:    %[[CONTRACT_1_1_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_1_0]], %[[READ_RHS_0_1]], %[[ACC_1_1]]
//      PROMOTE:    %[[CONTRACT_1_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_1_1]], %[[READ_RHS_1_1]], %[[CONTRACT_1_1_1]]
//      PROMOTE:    %[[CONTRACT_1_2_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_1_0]], %[[READ_RHS_0_2]], %[[ACC_1_2]]
//      PROMOTE:    %[[CONTRACT_1_2:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_1_1]], %[[READ_RHS_1_2]], %[[CONTRACT_1_2_1]]
//      PROMOTE:    %[[CONTRACT_1_3_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_1_0]], %[[READ_RHS_0_3]], %[[ACC_1_3]]
//      PROMOTE:    %[[CONTRACT_1_3:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_1_1]], %[[READ_RHS_1_3]], %[[CONTRACT_1_3_1]]

//      PROMOTE:    %[[CONTRACT_2_0_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_2_0]], %[[READ_RHS_0_0]], %[[ACC_2_0]]
//      PROMOTE:    %[[CONTRACT_2_0:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_2_1]], %[[READ_RHS_1_0]], %[[CONTRACT_2_0_1]]
//      PROMOTE:    %[[CONTRACT_2_1_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_2_0]], %[[READ_RHS_0_1]], %[[ACC_2_1]]
//      PROMOTE:    %[[CONTRACT_2_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_2_1]], %[[READ_RHS_1_1]], %[[CONTRACT_2_1_1]]
//      PROMOTE:    %[[CONTRACT_2_2_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_2_0]], %[[READ_RHS_0_2]], %[[ACC_2_2]]
//      PROMOTE:    %[[CONTRACT_2_2:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_2_1]], %[[READ_RHS_1_2]], %[[CONTRACT_2_2_1]]
//      PROMOTE:    %[[CONTRACT_2_3_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_2_0]], %[[READ_RHS_0_3]], %[[ACC_2_3]]
//      PROMOTE:    %[[CONTRACT_2_3:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_2_1]], %[[READ_RHS_1_3]], %[[CONTRACT_2_3_1]]

//      PROMOTE:    %[[CONTRACT_3_0_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_3_0]], %[[READ_RHS_0_0]], %[[ACC_3_0]]
//      PROMOTE:    %[[CONTRACT_3_0:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_3_1]], %[[READ_RHS_1_0]], %[[CONTRACT_3_0_1]]
//      PROMOTE:    %[[CONTRACT_3_1_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_3_0]], %[[READ_RHS_0_1]], %[[ACC_3_1]]
//      PROMOTE:    %[[CONTRACT_3_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_3_1]], %[[READ_RHS_1_1]], %[[CONTRACT_3_1_1]]
//      PROMOTE:    %[[CONTRACT_3_2_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_3_0]], %[[READ_RHS_0_2]], %[[ACC_3_2]]
//      PROMOTE:    %[[CONTRACT_3_2:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_3_1]], %[[READ_RHS_1_2]], %[[CONTRACT_3_2_1]]
//      PROMOTE:    %[[CONTRACT_3_3_1:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_3_0]], %[[READ_RHS_0_3]], %[[ACC_3_3]]
//      PROMOTE:    %[[CONTRACT_3_3:.+]] = vector.contract
// PROMOTE-SAME:      %[[READ_LHS_3_1]], %[[READ_RHS_1_3]], %[[CONTRACT_3_3_1]]

//      PROMOTE:    scf.yield %[[CONTRACT_0_0]], %[[CONTRACT_0_1]],
// PROMOTE-SAME:      %[[CONTRACT_0_2]], %[[CONTRACT_0_3]], %[[CONTRACT_1_0]],
// PROMOTE-SAME:      %[[CONTRACT_1_1]], %[[CONTRACT_1_2]], %[[CONTRACT_1_3]],
// PROMOTE-SAME:      %[[CONTRACT_2_0]], %[[CONTRACT_2_1]], %[[CONTRACT_2_2]],
// PROMOTE-SAME:      %[[CONTRACT_2_3]], %[[CONTRACT_3_0]], %[[CONTRACT_3_1]],
// PROMOTE-SAME:      %[[CONTRACT_3_2]], %[[CONTRACT_3_3]]

//  PROMOTE-DAG:  vector.transfer_write %[[FOR_RES]]#0, %[[SG_RESULT_SUBVIEW]][%[[C0]], %[[C0]]]
//  PROMOTE-DAG:  vector.transfer_write %[[FOR_RES]]#1, %[[SG_RESULT_SUBVIEW]][%[[C0]], %[[C16]]]
//  PROMOTE-DAG:  vector.transfer_write %[[FOR_RES]]#2, %[[SG_RESULT_SUBVIEW]][%[[C0]], %[[C32]]]
//  PROMOTE-DAG:  vector.transfer_write %[[FOR_RES]]#3, %[[SG_RESULT_SUBVIEW]][%[[C0]], %[[C48]]]

//  PROMOTE-DAG:  vector.transfer_write %[[FOR_RES]]#4, %[[SG_RESULT_SUBVIEW]][%[[C16]], %[[C0]]]
//  PROMOTE-DAG:  vector.transfer_write %[[FOR_RES]]#5, %[[SG_RESULT_SUBVIEW]][%[[C16]], %[[C16]]]
//  PROMOTE-DAG:  vector.transfer_write %[[FOR_RES]]#6, %[[SG_RESULT_SUBVIEW]][%[[C16]], %[[C32]]]
//  PROMOTE-DAG:  vector.transfer_write %[[FOR_RES]]#7, %[[SG_RESULT_SUBVIEW]][%[[C16]], %[[C48]]]

//  PROMOTE-DAG:  vector.transfer_write %[[FOR_RES]]#8, %[[SG_RESULT_SUBVIEW]][%[[C32]], %[[C0]]]
//  PROMOTE-DAG:  vector.transfer_write %[[FOR_RES]]#9, %[[SG_RESULT_SUBVIEW]][%[[C32]], %[[C16]]]
//  PROMOTE-DAG:  vector.transfer_write %[[FOR_RES]]#10, %[[SG_RESULT_SUBVIEW]][%[[C32]], %[[C32]]]
//  PROMOTE-DAG:  vector.transfer_write %[[FOR_RES]]#11, %[[SG_RESULT_SUBVIEW]][%[[C32]], %[[C48]]]

//  PROMOTE-DAG:  vector.transfer_write %[[FOR_RES]]#12, %[[SG_RESULT_SUBVIEW]][%[[C48]], %[[C0]]]
//  PROMOTE-DAG:  vector.transfer_write %[[FOR_RES]]#13, %[[SG_RESULT_SUBVIEW]][%[[C48]], %[[C16]]]
//  PROMOTE-DAG:  vector.transfer_write %[[FOR_RES]]#14, %[[SG_RESULT_SUBVIEW]][%[[C48]], %[[C32]]]
//  PROMOTE-DAG:  vector.transfer_write %[[FOR_RES]]#15, %[[SG_RESULT_SUBVIEW]][%[[C48]], %[[C48]]] 
