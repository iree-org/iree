// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.variant(iree-spirv-tile-and-vectorize,canonicalize,cse))" %s | IreeFileCheck %s
// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.variant(iree-spirv-tile-and-vectorize,canonicalize,cse))" -iree-spirv-use-workgroup-memory %s | IreeFileCheck %s -check-prefix=PROMOTE

hal.executable @matmul_static_shape attributes {sym_visibility = "private"} {
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target="vulkan" {
    hal.executable.entry_point @matmul_static_shape attributes {
      interface = @io,
      ordinal = 0 : index
    }
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
      func @matmul_static_shape() {
        %c32 = constant 32 : index
        %c4096 = constant 4096 : index
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4096x4096xf16>
        %1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<4096x4096xf16>
        %2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4096x4096xf16>
        %3 = hal.interface.workgroup.size[0] : index
        %4 = hal.interface.workgroup.size[1] : index
        scf.for %arg0 = %c0 to %c4096 step %c32 {
          %5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%4]
          %6 = memref.subview %0[%5, %arg0] [64, 32] [1, 1] : memref<4096x4096xf16> to memref<64x32xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>
          %7 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%3]
          %8 = memref.subview %1[%arg0, %7] [32, 64] [1, 1] : memref<4096x4096xf16> to memref<32x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>
          %9 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%4]
          %10 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%3]
          %11 = memref.subview %2[%9, %10] [64, 64] [1, 1] : memref<4096x4096xf16> to memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>
          linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%6, %8 : memref<64x32xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>, memref<32x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>) outs(%11 : memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>)
        }
        return
      }
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 64)>
//      CHECK: func @matmul_static_shape
//  CHECK-DAG:  %[[CST:.+]] = constant 0.0
//  CHECK-DAG:  %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:  %[[C16:.+]] = constant 16 : index
//  CHECK-DAG:  %[[C32:.+]] = constant 32 : index
//  CHECK-DAG:  %[[C48:.+]] = constant 48 : index
//  CHECK-DAG:  %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0[%[[C0]]]
//  CHECK-DAG:  %[[ARG1:.+]] = hal.interface.binding.subspan @io::@arg1[%[[C0]]]
//  CHECK-DAG:  %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0[%[[C0]]]
//      CHECK:  %[[BIDX:.+]] = hal.interface.workgroup.size[0]
//      CHECK:  %[[BIDY:.+]] = hal.interface.workgroup.size[1]
//      CHECK:  %[[BOFFSET_Y:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//      CHECK:  %[[BOFFSET_X:.+]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//      CHECK:    %[[SUBVIEW_RESULT:.+]] = memref.subview %[[RET0]]
// CHECK-SAME:      [%[[BOFFSET_Y]], %[[BOFFSET_X]]] [64, 64]

//  CHECK-DAG:  %[[READ_INIT_0_0:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT]][%[[C0]], %[[C0]]]
//  CHECK-DAG:  %[[READ_INIT_0_1:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT]][%[[C0]], %[[C16]]]
//  CHECK-DAG:  %[[READ_INIT_0_2:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT]][%[[C0]], %[[C32]]]
//  CHECK-DAG:  %[[READ_INIT_0_3:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT]][%[[C0]], %[[C48]]]

//  CHECK-DAG:  %[[READ_INIT_1_0:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT]][%[[C16]], %[[C0]]]
//  CHECK-DAG:  %[[READ_INIT_1_1:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT]][%[[C16]], %[[C16]]]
//  CHECK-DAG:  %[[READ_INIT_1_2:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT]][%[[C16]], %[[C32]]]
//  CHECK-DAG:  %[[READ_INIT_1_3:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT]][%[[C16]], %[[C48]]]

//  CHECK-DAG:  %[[READ_INIT_2_0:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT]][%[[C32]], %[[C0]]]
//  CHECK-DAG:  %[[READ_INIT_2_1:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT]][%[[C32]], %[[C16]]]
//  CHECK-DAG:  %[[READ_INIT_2_2:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT]][%[[C32]], %[[C32]]]
//  CHECK-DAG:  %[[READ_INIT_2_3:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT]][%[[C32]], %[[C48]]]

//  CHECK-DAG:  %[[READ_INIT_3_0:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT]][%[[C48]], %[[C0]]]
//  CHECK-DAG:  %[[READ_INIT_3_1:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT]][%[[C48]], %[[C16]]]
//  CHECK-DAG:  %[[READ_INIT_3_2:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT]][%[[C48]], %[[C32]]]
//  CHECK-DAG:  %[[READ_INIT_3_3:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT]][%[[C48]], %[[C48]]]

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
//      CHECK:    %[[SUBVIEW_LHS:.+]] = memref.subview %[[ARG0]]
// CHECK-SAME:      [%[[BOFFSET_Y]], %[[IV0]]] [64, 32]
//      CHECK:    %[[SUBVIEW_RHS:.+]] = memref.subview %[[ARG1]]
// CHECK-SAME:      [%[[IV0]], %[[BOFFSET_X]]] [32, 64]

//  CHECK-DAG:    %[[READ_LHS_0_0:.+]] = vector.transfer_read %[[SUBVIEW_LHS]][%[[C0]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_0_1:.+]] = vector.transfer_read %[[SUBVIEW_LHS]][%[[C0]], %[[C16]]]

//  CHECK-DAG:    %[[READ_LHS_1_0:.+]] = vector.transfer_read %[[SUBVIEW_LHS]][%[[C16]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_1_1:.+]] = vector.transfer_read %[[SUBVIEW_LHS]][%[[C16]], %[[C16]]]

//  CHECK-DAG:    %[[READ_LHS_2_0:.+]] = vector.transfer_read %[[SUBVIEW_LHS]][%[[C32]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_2_1:.+]] = vector.transfer_read %[[SUBVIEW_LHS]][%[[C32]], %[[C16]]]

//  CHECK-DAG:    %[[READ_LHS_3_0:.+]] = vector.transfer_read %[[SUBVIEW_LHS]][%[[C48]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_3_1:.+]] = vector.transfer_read %[[SUBVIEW_LHS]][%[[C48]], %[[C16]]]

//  CHECK-DAG:    %[[READ_RHS_0_0:.+]] = vector.transfer_read %[[SUBVIEW_RHS]][%[[C0]], %[[C0]]]
//  CHECK-DAG:    %[[READ_RHS_0_1:.+]] = vector.transfer_read %[[SUBVIEW_RHS]][%[[C0]], %[[C16]]]
//  CHECK-DAG:    %[[READ_RHS_0_2:.+]] = vector.transfer_read %[[SUBVIEW_RHS]][%[[C0]], %[[C32]]]
//  CHECK-DAG:    %[[READ_RHS_0_3:.+]] = vector.transfer_read %[[SUBVIEW_RHS]][%[[C0]], %[[C48]]]

//  CHECK-DAG:    %[[READ_RHS_1_0:.+]] = vector.transfer_read %[[SUBVIEW_RHS]][%[[C16]], %[[C0]]]
//  CHECK-DAG:    %[[READ_RHS_1_1:.+]] = vector.transfer_read %[[SUBVIEW_RHS]][%[[C16]], %[[C16]]]
//  CHECK-DAG:    %[[READ_RHS_1_2:.+]] = vector.transfer_read %[[SUBVIEW_RHS]][%[[C16]], %[[C32]]]
//  CHECK-DAG:    %[[READ_RHS_1_3:.+]] = vector.transfer_read %[[SUBVIEW_RHS]][%[[C16]], %[[C48]]]

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

//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#0, %[[SUBVIEW_RESULT]][%[[C0]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#1, %[[SUBVIEW_RESULT]][%[[C0]], %[[C16]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#2, %[[SUBVIEW_RESULT]][%[[C0]], %[[C32]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#3, %[[SUBVIEW_RESULT]][%[[C0]], %[[C48]]]

//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#4, %[[SUBVIEW_RESULT]][%[[C16]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#5, %[[SUBVIEW_RESULT]][%[[C16]], %[[C16]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#6, %[[SUBVIEW_RESULT]][%[[C16]], %[[C32]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#7, %[[SUBVIEW_RESULT]][%[[C16]], %[[C48]]]

//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#8, %[[SUBVIEW_RESULT]][%[[C32]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#9, %[[SUBVIEW_RESULT]][%[[C32]], %[[C16]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#10, %[[SUBVIEW_RESULT]][%[[C32]], %[[C32]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#11, %[[SUBVIEW_RESULT]][%[[C32]], %[[C48]]]

//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#12, %[[SUBVIEW_RESULT]][%[[C48]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#13, %[[SUBVIEW_RESULT]][%[[C48]], %[[C16]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#14, %[[SUBVIEW_RESULT]][%[[C48]], %[[C32]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#15, %[[SUBVIEW_RESULT]][%[[C48]], %[[C48]]]

// -----

hal.executable @matmul_static_shape attributes {sym_visibility = "private"} {
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target="vulkan" {
    hal.executable.entry_point @matmul_static_shape attributes {
      interface = @io,
      ordinal = 0 : index
    }
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
      func @matmul_static_shape() {
        %c32 = constant 32 : index
        %c4096 = constant 4096 : index
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4096x4096xf16>
        %1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<4096x4096xf16>
        %2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4096x4096xf16>
        %3 = hal.interface.workgroup.size[0] : index
        %4 = hal.interface.workgroup.size[1] : index
        scf.for %arg0 = %c0 to %c4096 step %c32 {
          %5 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%4]
          %6 = memref.subview %0[%5, %arg0] [128, 32] [1, 1] : memref<4096x4096xf16> to memref<128x32xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>
          %7 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%3]
          %8 = memref.subview %1[%arg0, %7] [32, 128] [1, 1] : memref<4096x4096xf16> to memref<32x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>
          %9 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%4]
          %10 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%3]
          %11 = memref.subview %2[%9, %10] [128, 128] [1, 1] : memref<4096x4096xf16> to memref<128x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>
          linalg.matmul {__internal_linalg_transform__ = "workgroup", is_root_op, launch_info_key = "__op_num_0__"} ins(%6, %8 : memref<128x32xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>, memref<32x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>) outs(%11 : memref<128x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>)
        }
        return
      }
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

//  PROMOTE-DAG: #[[MAP4:.+]] = affine_map<()[s0] -> (s0 * 64 - (s0 floordiv 2) * 128)>
//      PROMOTE: func @matmul_static_shape
//  PROMOTE-DAG:  %[[C0:.+]] = constant 0 : index
//  PROMOTE-DAG:  %[[C2:.+]] = constant 2
//  PROMOTE-DAG:  %[[C16:.+]] = constant 16
//  PROMOTE-DAG:  %[[C32:.+]] = constant 32
//  PROMOTE-DAG:  %[[C48:.+]] = constant 48
//  PROMOTE-DAG:  %[[ALLOC1:.+]] = memref.alloc() : memref<128x32xf16, 3>
//  PROMOTE-DAG:  %[[ALLOC2:.+]] = memref.alloc() : memref<32x128xf16, 3>
//  PROMOTE-DAG:  %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0[%[[C0]]]
//  PROMOTE-DAG:  %[[ARG1:.+]] = hal.interface.binding.subspan @io::@arg1[%[[C0]]]
//  PROMOTE-DAG:  %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0[%[[C0]]]

//      PROMOTE:  %[[RESULT_SUBVIEW:.+]] = memref.subview %[[RET0]]
//      PROMOTE:  %[[WGMEM_LHS_SUBVIEW:.+]] = memref.subview %[[ALLOC1]][0, 0] [128, 32] [1, 1]
//      PROMOTE:  %[[WGMEM_RHS_SUBVIEW:.+]] = memref.subview %[[ALLOC2]][0, 0] [32, 128] [1, 1]
//      PROMOTE:  %[[SG_X:.+]] = gpu.subgroup_id
//      PROMOTE:  %[[SG_Y:.+]] = divi_signed %[[SG_X]], %[[C2]]
//      PROMOTE:  %[[SGOFFSET_Y:.+]] = affine.apply #[[MAP4]]()[%[[SG_Y]]]
//      PROMOTE:  %[[SG_LHS_SUBVIEW:.+]] = memref.subview %[[WGMEM_LHS_SUBVIEW]][%[[SGOFFSET_Y]], 0]
//      PROMOTE:  %[[SGOFFSET_X:.+]] = affine.apply #[[MAP4]]()[%[[SG_X]]]
//      PROMOTE:  %[[SG_RHS_SUBVIEW:.+]] = memref.subview %[[WGMEM_RHS_SUBVIEW]][0, %[[SGOFFSET_X]]]
//      PROMOTE:  %[[SG_RESULT_SUBVIEW:.+]] = memref.subview %[[RESULT_SUBVIEW]][%[[SGOFFSET_Y]], %[[SGOFFSET_X]]]

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

//      PROMOTE:    %[[LHS_SUBVIEW:.+]] = memref.subview %[[ARG0]]
//      PROMOTE:    %[[RHS_SUBVIEW:.+]] = memref.subview %[[ARG1]]
//      PROMOTE:    linalg.copy(%[[LHS_SUBVIEW]], %[[WGMEM_LHS_SUBVIEW]])
//      PROMOTE:    linalg.copy(%[[RHS_SUBVIEW]], %[[WGMEM_RHS_SUBVIEW]])

//  PROMOTE-DAG:    %[[READ_LHS_0_0:.+]] = vector.transfer_read %[[SG_LHS_SUBVIEW]][%[[C0]], %[[C0]]]
//  PROMOTE-DAG:    %[[READ_LHS_0_1:.+]] = vector.transfer_read %[[SG_LHS_SUBVIEW]][%[[C0]], %[[C16]]]

//  PROMOTE-DAG:    %[[READ_LHS_1_0:.+]] = vector.transfer_read %[[SG_LHS_SUBVIEW]][%[[C16]], %[[C0]]]
//  PROMOTE-DAG:    %[[READ_LHS_1_1:.+]] = vector.transfer_read %[[SG_LHS_SUBVIEW]][%[[C16]], %[[C16]]]

//  PROMOTE-DAG:    %[[READ_LHS_2_0:.+]] = vector.transfer_read %[[SG_LHS_SUBVIEW]][%[[C32]], %[[C0]]]
//  PROMOTE-DAG:    %[[READ_LHS_2_1:.+]] = vector.transfer_read %[[SG_LHS_SUBVIEW]][%[[C32]], %[[C16]]]

//  PROMOTE-DAG:    %[[READ_LHS_3_0:.+]] = vector.transfer_read %[[SG_LHS_SUBVIEW]][%[[C48]], %[[C0]]]
//  PROMOTE-DAG:    %[[READ_LHS_3_1:.+]] = vector.transfer_read %[[SG_LHS_SUBVIEW]][%[[C48]], %[[C16]]]

//  PROMOTE-DAG:    %[[READ_RHS_0_0:.+]] = vector.transfer_read %[[SG_RHS_SUBVIEW]][%[[C0]], %[[C0]]]
//  PROMOTE-DAG:    %[[READ_RHS_0_1:.+]] = vector.transfer_read %[[SG_RHS_SUBVIEW]][%[[C0]], %[[C16]]]
//  PROMOTE-DAG:    %[[READ_RHS_0_2:.+]] = vector.transfer_read %[[SG_RHS_SUBVIEW]][%[[C0]], %[[C32]]]
//  PROMOTE-DAG:    %[[READ_RHS_0_3:.+]] = vector.transfer_read %[[SG_RHS_SUBVIEW]][%[[C0]], %[[C48]]]

//  PROMOTE-DAG:    %[[READ_RHS_1_0:.+]] = vector.transfer_read %[[SG_RHS_SUBVIEW]][%[[C16]], %[[C0]]]
//  PROMOTE-DAG:    %[[READ_RHS_1_1:.+]] = vector.transfer_read %[[SG_RHS_SUBVIEW]][%[[C16]], %[[C16]]]
//  PROMOTE-DAG:    %[[READ_RHS_1_2:.+]] = vector.transfer_read %[[SG_RHS_SUBVIEW]][%[[C16]], %[[C32]]]
//  PROMOTE-DAG:    %[[READ_RHS_1_3:.+]] = vector.transfer_read %[[SG_RHS_SUBVIEW]][%[[C16]], %[[C48]]]

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
