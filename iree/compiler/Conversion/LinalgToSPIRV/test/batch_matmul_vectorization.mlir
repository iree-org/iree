// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-linalg-tile-and-fuse,canonicalize,cse))" -iree-spirv-enable-vectorization %s | IreeFileCheck %s

hal.executable @batch_matmul_static_shape attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @batch_matmul_static_shape attributes {
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
      func @batch_matmul_static_shape()
        attributes {vkspv.num_workgroups_fn = @matmul_static_shape__num_workgroups__} {
        %arg0 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@arg0, operand_result_num = 0 : i32} : memref<4x1024x1024xf32>
        %arg1 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@arg1, operand_result_num = 1 : i32} : memref<4x1024x1024xf32>
        %ret0 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@ret0, operand_result_num = 2 : i32} : memref<4x1024x1024xf32>
        linalg.batch_matmul ins(%arg0, %arg1 : memref<4x1024x1024xf32>, memref<4x1024x1024xf32>) outs(%ret0 : memref<4x1024x1024xf32>)
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
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 64)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 4)>
//      CHECK: func @batch_matmul_static_shape
//  CHECK-DAG:  %[[ARG0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg0
//  CHECK-DAG:  %[[ARG1:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg1
//  CHECK-DAG:  %[[RET0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@ret0
//  CHECK-DAG:  %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:  %[[CST:.+]] = constant 0.0
//  CHECK-DAG:  %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:  %[[C2:.+]] = constant 2 : index
//  CHECK-DAG:  %[[C3:.+]] = constant 3 : index
//  CHECK-DAG:  %[[C4:.+]] = constant 4 : index
//  CHECK-DAG:  %[[C5:.+]] = constant 5 : index
//  CHECK-DAG:  %[[C6:.+]] = constant 6 : index
//  CHECK-DAG:  %[[C7:.+]] = constant 7 : index
//      CHECK:  %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//      CHECK:  %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//      CHECK:  %[[BIDZ:.+]] = "gpu.block_id"() {dimension = "z"}
//      CHECK:  %[[BOFFSET_Y:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//      CHECK:  %[[BOFFSET_X:.+]] = affine.apply #[[MAP1]]()[%[[BIDX]]]
//      CHECK:  %[[SUBVIEW_RESULT:.+]] = subview %[[RET0]]
// CHECK-SAME:      [%[[BIDZ]], %[[BOFFSET_Y]], %[[BOFFSET_X]]] [1, 8, 64]
//      CHECK:  %[[IIDX:.+]] = "gpu.thread_id"() {dimension = "x"}
//      CHECK:  %[[IIDY:.+]] = "gpu.thread_id"() {dimension = "y"}
//      CHECK:  %[[IIDZ:.+]] = "gpu.thread_id"() {dimension = "z"}
//      CHECK:  %[[IOFFSET_Y:.+]] = affine.apply #[[MAP0]]()[%[[IIDY]]]
//      CHECK:  %[[IOFFSET_X:.+]] = affine.apply #[[MAP2]]()[%[[IIDX]]]
//      CHECK:  %[[SUBVIEW_RESULT_2:.+]] = subview %[[SUBVIEW_RESULT]]
// CHECK-SAME:      [%[[IIDZ]], %[[IOFFSET_Y]], %[[IOFFSET_X]]] [1, 8, 4]
//  CHECK-DAG:  %[[READ_INIT_0:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C0]],  %[[C0]]]
//  CHECK-DAG:  %[[READ_INIT_1:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C1]],  %[[C0]]]
//  CHECK-DAG:  %[[READ_INIT_2:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C2]],  %[[C0]]]
//  CHECK-DAG:  %[[READ_INIT_3:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C3]],  %[[C0]]]
//  CHECK-DAG:  %[[READ_INIT_4:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C4]],  %[[C0]]]
//  CHECK-DAG:  %[[READ_INIT_5:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C5]],  %[[C0]]]
//  CHECK-DAG:  %[[READ_INIT_6:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C6]],  %[[C0]]]
//  CHECK-DAG:  %[[READ_INIT_7:.+]] = vector.transfer_read
// CHECK-SAME:    %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C7]],  %[[C0]]]

//      CHECK:  %[[FOR_RES:.+]]:8 = scf.for %[[IV0:.+]] = {{.*}} to
// CHECK-SAME:  iter_args(%[[ACC_0:.+]] = %[[READ_INIT_0]],
// CHECK-SAME:  %[[ACC_1:.+]] = %[[READ_INIT_1]],
// CHECK-SAME:  %[[ACC_2:.+]] = %[[READ_INIT_2]],
// CHECK-SAME:  %[[ACC_3:.+]] = %[[READ_INIT_3]],
// CHECK-SAME:  %[[ACC_4:.+]] = %[[READ_INIT_4]],
// CHECK-SAME:  %[[ACC_5:.+]] = %[[READ_INIT_5]],
// CHECK-SAME:  %[[ACC_6:.+]] = %[[READ_INIT_6]],
// CHECK-SAME:  %[[ACC_7:.+]] = %[[READ_INIT_7]])
//      CHECK:    %[[SUBVIEW_LHS:.+]] = subview %[[ARG0]]
// CHECK-SAME:      [%[[BIDZ]], %[[BOFFSET_Y]], %[[IV0]]] [1, 8, 4]
//      CHECK:    %[[SUBVIEW_RHS:.+]] = subview %[[ARG1]]
// CHECK-SAME:      [%[[BIDZ]], %[[IV0]], %[[BOFFSET_X]]] [1, 4, 64]
//      CHECK:    %[[SUBVIEW_LHS_2:.+]] = subview %[[SUBVIEW_LHS]]
// CHECK-SAME:      [%[[IIDZ]], %[[IOFFSET_Y]], 0] [1, 8, 4] [1, 1, 1]
//      CHECK:    %[[SUBVIEW_RHS_2:.+]] = subview %[[SUBVIEW_RHS]]
// CHECK-SAME:      [%[[IIDZ]], 0, %[[IOFFSET_X]]] [1, 4, 4] [1, 1, 1]

//  CHECK-DAG:    %[[READ_LHS_0:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_LHS_2]][%[[C0]], %[[C0]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_1:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_LHS_2]][%[[C0]], %[[C1]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_2:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_LHS_2]][%[[C0]], %[[C2]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_3:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_LHS_2]][%[[C0]], %[[C3]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_4:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_LHS_2]][%[[C0]], %[[C4]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_5:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_LHS_2]][%[[C0]], %[[C5]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_6:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_LHS_2]][%[[C0]], %[[C6]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_7:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_LHS_2]][%[[C0]], %[[C7]], %[[C0]]]

//  CHECK-DAG:    %[[READ_RHS_0:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_RHS_2]][%[[C0]], %[[C0]], %[[C0]]]
//  CHECK-DAG:    %[[READ_RHS_1:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_RHS_2]][%[[C0]], %[[C1]], %[[C0]]]
//  CHECK-DAG:    %[[READ_RHS_2:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_RHS_2]][%[[C0]], %[[C2]], %[[C0]]]
//  CHECK-DAG:    %[[READ_RHS_3:.+]] = vector.transfer_read
// CHECK-SAME:      %[[SUBVIEW_RHS_2]][%[[C0]], %[[C3]], %[[C0]]]

//  CHECK-DAG:    %[[READ_LHS_0_0:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_0]] {offsets = [0, 0, 0]
//  CHECK-DAG:    %[[READ_LHS_0_1:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_0]] {offsets = [0, 0, 1]
//  CHECK-DAG:    %[[READ_LHS_0_2:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_0]] {offsets = [0, 0, 2]
//  CHECK-DAG:    %[[READ_LHS_0_3:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_0]] {offsets = [0, 0, 3]
//  CHECK-DAG:    %[[READ_LHS_1_0:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_1]] {offsets = [0, 0, 0]
//  CHECK-DAG:    %[[READ_LHS_1_1:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_1]] {offsets = [0, 0, 1]
//  CHECK-DAG:    %[[READ_LHS_1_2:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_1]] {offsets = [0, 0, 2]
//  CHECK-DAG:    %[[READ_LHS_1_3:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_1]] {offsets = [0, 0, 3]
//  CHECK-DAG:    %[[READ_LHS_2_0:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_2]] {offsets = [0, 0, 0]
//  CHECK-DAG:    %[[READ_LHS_2_1:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_2]] {offsets = [0, 0, 1]
//  CHECK-DAG:    %[[READ_LHS_2_2:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_2]] {offsets = [0, 0, 2]
//  CHECK-DAG:    %[[READ_LHS_2_3:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_2]] {offsets = [0, 0, 3]
//  CHECK-DAG:    %[[READ_LHS_3_0:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_3]] {offsets = [0, 0, 0]
//  CHECK-DAG:    %[[READ_LHS_3_1:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_3]] {offsets = [0, 0, 1]
//  CHECK-DAG:    %[[READ_LHS_3_2:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_3]] {offsets = [0, 0, 2]
//  CHECK-DAG:    %[[READ_LHS_3_3:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_3]] {offsets = [0, 0, 3]
//  CHECK-DAG:    %[[READ_LHS_4_0:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_4]] {offsets = [0, 0, 0]
//  CHECK-DAG:    %[[READ_LHS_4_1:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_4]] {offsets = [0, 0, 1]
//  CHECK-DAG:    %[[READ_LHS_4_2:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_4]] {offsets = [0, 0, 2]
//  CHECK-DAG:    %[[READ_LHS_4_3:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_4]] {offsets = [0, 0, 3]
//  CHECK-DAG:    %[[READ_LHS_5_0:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_5]] {offsets = [0, 0, 0]
//  CHECK-DAG:    %[[READ_LHS_5_1:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_5]] {offsets = [0, 0, 1]
//  CHECK-DAG:    %[[READ_LHS_5_2:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_5]] {offsets = [0, 0, 2]
//  CHECK-DAG:    %[[READ_LHS_5_3:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_5]] {offsets = [0, 0, 3]
//  CHECK-DAG:    %[[READ_LHS_6_0:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_6]] {offsets = [0, 0, 0]
//  CHECK-DAG:    %[[READ_LHS_6_1:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_6]] {offsets = [0, 0, 1]
//  CHECK-DAG:    %[[READ_LHS_6_2:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_6]] {offsets = [0, 0, 2]
//  CHECK-DAG:    %[[READ_LHS_6_3:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_6]] {offsets = [0, 0, 3]
//  CHECK-DAG:    %[[READ_LHS_7_0:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_7]] {offsets = [0, 0, 0]
//  CHECK-DAG:    %[[READ_LHS_7_1:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_7]] {offsets = [0, 0, 1]
//  CHECK-DAG:    %[[READ_LHS_7_2:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_7]] {offsets = [0, 0, 2]
//  CHECK-DAG:    %[[READ_LHS_7_3:.+]] = vector.extract_strided_slice
// CHECK-SAME:      %[[READ_LHS_7]] {offsets = [0, 0, 3]

//      CHECK:    %[[CONTRACT_0_0:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_0_0]], %[[READ_RHS_0]], %[[ACC_0]]
//      CHECK:    %[[CONTRACT_0_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_0_1]], %[[READ_RHS_1]], %[[CONTRACT_0_0]]
//      CHECK:    %[[CONTRACT_0_2:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_0_2]], %[[READ_RHS_2]], %[[CONTRACT_0_1]]
//      CHECK:    %[[CONTRACT_0_3:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_0_3]], %[[READ_RHS_3]], %[[CONTRACT_0_2]]

//      CHECK:    %[[CONTRACT_1_0:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_1_0]], %[[READ_RHS_0]], %[[ACC_1]]
//      CHECK:    %[[CONTRACT_1_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_1_1]], %[[READ_RHS_1]], %[[CONTRACT_1_0]]
//      CHECK:    %[[CONTRACT_1_2:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_1_2]], %[[READ_RHS_2]], %[[CONTRACT_1_1]]
//      CHECK:    %[[CONTRACT_1_3:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_1_3]], %[[READ_RHS_3]], %[[CONTRACT_1_2]]

//      CHECK:    %[[CONTRACT_2_0:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_2_0]], %[[READ_RHS_0]], %[[ACC_2]]
//      CHECK:    %[[CONTRACT_2_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_2_1]], %[[READ_RHS_1]], %[[CONTRACT_2_0]]
//      CHECK:    %[[CONTRACT_2_2:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_2_2]], %[[READ_RHS_2]], %[[CONTRACT_2_1]]
//      CHECK:    %[[CONTRACT_2_3:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_2_3]], %[[READ_RHS_3]], %[[CONTRACT_2_2]]

//      CHECK:    %[[CONTRACT_3_0:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_3_0]], %[[READ_RHS_0]], %[[ACC_3]]
//      CHECK:    %[[CONTRACT_3_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_3_1]], %[[READ_RHS_1]], %[[CONTRACT_3_0]]
//      CHECK:    %[[CONTRACT_3_2:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_3_2]], %[[READ_RHS_2]], %[[CONTRACT_3_1]]
//      CHECK:    %[[CONTRACT_3_3:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_3_3]], %[[READ_RHS_3]], %[[CONTRACT_3_2]]

//      CHECK:    %[[CONTRACT_4_0:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_4_0]], %[[READ_RHS_0]], %[[ACC_4]]
//      CHECK:    %[[CONTRACT_4_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_4_1]], %[[READ_RHS_1]], %[[CONTRACT_4_0]]
//      CHECK:    %[[CONTRACT_4_2:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_4_2]], %[[READ_RHS_2]], %[[CONTRACT_4_1]]
//      CHECK:    %[[CONTRACT_4_3:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_4_3]], %[[READ_RHS_3]], %[[CONTRACT_4_2]]

//      CHECK:    %[[CONTRACT_5_0:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_5_0]], %[[READ_RHS_0]], %[[ACC_5]]
//      CHECK:    %[[CONTRACT_5_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_5_1]], %[[READ_RHS_1]], %[[CONTRACT_5_0]]
//      CHECK:    %[[CONTRACT_5_2:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_5_2]], %[[READ_RHS_2]], %[[CONTRACT_5_1]]
//      CHECK:    %[[CONTRACT_5_3:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_5_3]], %[[READ_RHS_3]], %[[CONTRACT_5_2]]

//      CHECK:    %[[CONTRACT_6_0:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_6_0]], %[[READ_RHS_0]], %[[ACC_6]]
//      CHECK:    %[[CONTRACT_6_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_6_1]], %[[READ_RHS_1]], %[[CONTRACT_6_0]]
//      CHECK:    %[[CONTRACT_6_2:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_6_2]], %[[READ_RHS_2]], %[[CONTRACT_6_1]]
//      CHECK:    %[[CONTRACT_6_3:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_6_3]], %[[READ_RHS_3]], %[[CONTRACT_6_2]]

//      CHECK:    %[[CONTRACT_7_0:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_7_0]], %[[READ_RHS_0]], %[[ACC_7]]
//      CHECK:    %[[CONTRACT_7_1:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_7_1]], %[[READ_RHS_1]], %[[CONTRACT_7_0]]
//      CHECK:    %[[CONTRACT_7_2:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_7_2]], %[[READ_RHS_2]], %[[CONTRACT_7_1]]
//      CHECK:    %[[CONTRACT_7_3:.+]] = vector.contract
// CHECK-SAME:      %[[READ_LHS_7_3]], %[[READ_RHS_3]], %[[CONTRACT_7_2]]

//      CHECK:  scf.yield %[[CONTRACT_0_3]], %[[CONTRACT_1_3]],
// CHECK-SAME:    %[[CONTRACT_2_3]], %[[CONTRACT_3_3]], %[[CONTRACT_4_3]],
// CHECK-SAME:    %[[CONTRACT_5_3]], %[[CONTRACT_6_3]], %[[CONTRACT_7_3]]

//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#0, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C0]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#1, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C1]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#2, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C2]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#3, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C3]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#4, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C4]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#5, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C5]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#6, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C6]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#7, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C7]], %[[C0]]]

// -----

hal.executable @batch_matmul_fused_fillop attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @batch_matmul_fused_fillop attributes {
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
      func @batch_matmul_fused_fillop()
        attributes {vkspv.num_workgroups_fn = @batch_matmul_fused_fillop__num_workgroups__} {
        %cst = constant 0.000000e+00 : f32
        %arg0 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@arg0, operand_result_num = 0 : i32} : memref<4x1024x1024xf32>
        %arg1 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@arg1, operand_result_num = 1 : i32} : memref<4x1024x1024xf32>
        %ret0 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@ret0, operand_result_num = 2 : i32} : memref<4x1024x1024xf32>
        linalg.fill(%ret0, %cst) : memref<4x1024x1024xf32>, f32
        linalg.batch_matmul ins(%arg0, %arg1 : memref<4x1024x1024xf32>, memref<4x1024x1024xf32>) outs(%ret0 : memref<4x1024x1024xf32>)
        return
      }
      func private @batch_matmul_fused_fillop__num_workgroups__
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
//    CHECK-LABEL: func @batch_matmul_fused_fillop
//  CHECK-COUNT-8:   vector.transfer_write
//  CHECK-COUNT-8:   vector.transfer_read
//          CHECK:   %[[FOR_RES:.+]]:8 = scf.for
// CHECK-COUNT-12:     vector.transfer_read
// CHECK-COUNT-32:     vector.contract
//      CHECK:         scf.yield
//  CHECK-COUNT-8:    vector.transfer_write %[[FOR_RES]]
//          CHECK:    return
