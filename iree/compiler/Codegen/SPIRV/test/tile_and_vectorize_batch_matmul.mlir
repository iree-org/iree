// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.variant(iree-spirv-concretize-workgroup-tiles,iree-spirv-tile-and-vectorize))" -canonicalize -cse -iree-spirv-workgroup-tile-size=1,8,64,4 -iree-spirv-invocation-tile-size=1,8,4,4 -iree-spirv-workgroup-size=16,1,1 %s | IreeFileCheck %s

hal.executable @batch_matmul_static_shape attributes {sym_visibility = "private"} {
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan_spirv, filter="vulkan*" {
    hal.executable.entry_point @batch_matmul_static_shape attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, ARM:IntegratedGPU, {}>}  {
      func @batch_matmul_static_shape() {
        %c0 = constant 0 : index
        %c4 = constant 4 : index
        %c1024 = constant 1024 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<4x1024x1024xf32>
        %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : memref<4x1024x1024xf32>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : memref<4x1024x1024xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_z, %workgroup_size_z]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_z, %workgroup_size_z]
        scf.for %arg0 = %3 to %c4 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c1024 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c1024 step %8 {
              %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4)>(%arg0)[%workgroup_size_z]
              %10 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1024)>(%arg1)[%workgroup_size_y]
              %11 = memref.subview %0[%arg0, %arg1, 0] [%9, %10, 1024] [1, 1, 1] : memref<4x1024x1024xf32> to memref<?x?x1024xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
              %12 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1024)>(%arg2)[%workgroup_size_x]
              %13 = memref.subview %1[%arg0, 0, %arg2] [%9, 1024, %12] [1, 1, 1] : memref<4x1024x1024xf32> to memref<?x1024x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
              %14 = memref.subview %2[%arg0, %arg1, %arg2] [%9, %10, %12] [1, 1, 1] : memref<4x1024x1024xf32> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
              linalg.batch_matmul {__internal_linalg_transform__ = "workgroup"} ins(%11, %13 : memref<?x?x1024xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>, memref<?x1024x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>) outs(%14 : memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>)
            }
          }
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

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 64)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 4)>
//      CHECK: func @batch_matmul_static_shape
//  CHECK-DAG:  %[[ARG0:.+]] = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0]
//  CHECK-DAG:  %[[ARG1:.+]] = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0]
//  CHECK-DAG:  %[[RET0:.+]] = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0]
//  CHECK-DAG:  %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:  %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:  %[[C2:.+]] = constant 2 : index
//  CHECK-DAG:  %[[C3:.+]] = constant 3 : index
//  CHECK-DAG:  %[[C4:.+]] = constant 4 : index
//  CHECK-DAG:  %[[C5:.+]] = constant 5 : index
//  CHECK-DAG:  %[[C6:.+]] = constant 6 : index
//  CHECK-DAG:  %[[C7:.+]] = constant 7 : index
//      CHECK:  %[[BIDX:.+]] = hal.interface.workgroup.id[0]
//      CHECK:  %[[BIDY:.+]] = hal.interface.workgroup.id[1]
//      CHECK:  %[[BIDZ:.+]] = hal.interface.workgroup.id[2]
//  CHECK-DAG:  %[[BOFFSET_Y:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//  CHECK-DAG:  %[[BOFFSET_X:.+]] = affine.apply #[[MAP1]]()[%[[BIDX]]]
//      CHECK:  %[[SUBVIEW_ARG0:.+]] = memref.subview %[[ARG0]]
// CHECK-SAME:      [%[[BIDZ]], %[[BOFFSET_Y]], 0] [1, 8, 1024]
//      CHECK:  %[[SUBVIEW_ARG1:.+]] = memref.subview %[[ARG1]]
// CHECK-SAME:      [%[[BIDZ]], 0, %[[BOFFSET_X]]] [1, 1024, 64]
//      CHECK:  %[[SUBVIEW_RESULT:.+]] = memref.subview %[[RET0]]
// CHECK-SAME:      [%[[BIDZ]], %[[BOFFSET_Y]], %[[BOFFSET_X]]] [1, 8, 64]
//      CHECK:  %[[IIDX:.+]] = "gpu.thread_id"() {dimension = "x"}
//      CHECK:  %[[IIDY:.+]] = "gpu.thread_id"() {dimension = "y"}
//      CHECK:  %[[IIDZ:.+]] = "gpu.thread_id"() {dimension = "z"}
//  CHECK-DAG:  %[[IOFFSET_Y:.+]] = affine.apply #[[MAP0]]()[%[[IIDY]]]
//  CHECK-DAG:  %[[IOFFSET_X:.+]] = affine.apply #[[MAP2]]()[%[[IIDX]]]
//      CHECK:  %[[SUBVIEW_RESULT_2:.+]] = memref.subview %[[SUBVIEW_RESULT]]
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
//  CHECK-DAG:    %[[SUBVIEW_LHS:.+]] = memref.subview %[[SUBVIEW_ARG0]]
// CHECK-SAME:      [%[[IIDZ]], %[[IOFFSET_Y]], %[[IV0]]] [1, 8, 4]
//  CHECK-DAG:    %[[SUBVIEW_RHS:.+]] = memref.subview %[[SUBVIEW_ARG1]]
// CHECK-SAME:      [%[[IIDZ]], %[[IV0]], %[[IOFFSET_X]]] [1, 4, 4] [1, 1, 1]

//  CHECK-DAG:    %[[READ_LHS_0:.+]] = vector.transfer_read %[[SUBVIEW_LHS]][%[[C0]], %[[C0]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_1:.+]] = vector.transfer_read %[[SUBVIEW_LHS]][%[[C0]], %[[C1]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_2:.+]] = vector.transfer_read %[[SUBVIEW_LHS]][%[[C0]], %[[C2]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_3:.+]] = vector.transfer_read %[[SUBVIEW_LHS]][%[[C0]], %[[C3]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_4:.+]] = vector.transfer_read %[[SUBVIEW_LHS]][%[[C0]], %[[C4]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_5:.+]] = vector.transfer_read %[[SUBVIEW_LHS]][%[[C0]], %[[C5]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_6:.+]] = vector.transfer_read %[[SUBVIEW_LHS]][%[[C0]], %[[C6]], %[[C0]]]
//  CHECK-DAG:    %[[READ_LHS_7:.+]] = vector.transfer_read %[[SUBVIEW_LHS]][%[[C0]], %[[C7]], %[[C0]]]

//  CHECK-DAG:    %[[READ_RHS_0:.+]] = vector.transfer_read %[[SUBVIEW_RHS]][%[[C0]], %[[C0]], %[[C0]]]
//  CHECK-DAG:    %[[READ_RHS_1:.+]] = vector.transfer_read %[[SUBVIEW_RHS]][%[[C0]], %[[C1]], %[[C0]]]
//  CHECK-DAG:    %[[READ_RHS_2:.+]] = vector.transfer_read %[[SUBVIEW_RHS]][%[[C0]], %[[C2]], %[[C0]]]
//  CHECK-DAG:    %[[READ_RHS_3:.+]] = vector.transfer_read %[[SUBVIEW_RHS]][%[[C0]], %[[C3]], %[[C0]]]

//  CHECK-DAG:    %[[READ_LHS_0_0:.+]] = vector.extract_strided_slice %[[READ_LHS_0]] {offsets = [0, 0, 0]
//  CHECK-DAG:    %[[READ_LHS_0_1:.+]] = vector.extract_strided_slice %[[READ_LHS_0]] {offsets = [0, 0, 1]
//  CHECK-DAG:    %[[READ_LHS_0_2:.+]] = vector.extract_strided_slice %[[READ_LHS_0]] {offsets = [0, 0, 2]
//  CHECK-DAG:    %[[READ_LHS_0_3:.+]] = vector.extract_strided_slice %[[READ_LHS_0]] {offsets = [0, 0, 3]
//  CHECK-DAG:    %[[READ_LHS_1_0:.+]] = vector.extract_strided_slice %[[READ_LHS_1]] {offsets = [0, 0, 0]
//  CHECK-DAG:    %[[READ_LHS_1_1:.+]] = vector.extract_strided_slice %[[READ_LHS_1]] {offsets = [0, 0, 1]
//  CHECK-DAG:    %[[READ_LHS_1_2:.+]] = vector.extract_strided_slice %[[READ_LHS_1]] {offsets = [0, 0, 2]
//  CHECK-DAG:    %[[READ_LHS_1_3:.+]] = vector.extract_strided_slice %[[READ_LHS_1]] {offsets = [0, 0, 3]
//  CHECK-DAG:    %[[READ_LHS_2_0:.+]] = vector.extract_strided_slice %[[READ_LHS_2]] {offsets = [0, 0, 0]
//  CHECK-DAG:    %[[READ_LHS_2_1:.+]] = vector.extract_strided_slice %[[READ_LHS_2]] {offsets = [0, 0, 1]
//  CHECK-DAG:    %[[READ_LHS_2_2:.+]] = vector.extract_strided_slice %[[READ_LHS_2]] {offsets = [0, 0, 2]
//  CHECK-DAG:    %[[READ_LHS_2_3:.+]] = vector.extract_strided_slice %[[READ_LHS_2]] {offsets = [0, 0, 3]
//  CHECK-DAG:    %[[READ_LHS_3_0:.+]] = vector.extract_strided_slice %[[READ_LHS_3]] {offsets = [0, 0, 0]
//  CHECK-DAG:    %[[READ_LHS_3_1:.+]] = vector.extract_strided_slice %[[READ_LHS_3]] {offsets = [0, 0, 1]
//  CHECK-DAG:    %[[READ_LHS_3_2:.+]] = vector.extract_strided_slice %[[READ_LHS_3]] {offsets = [0, 0, 2]
//  CHECK-DAG:    %[[READ_LHS_3_3:.+]] = vector.extract_strided_slice %[[READ_LHS_3]] {offsets = [0, 0, 3]
//  CHECK-DAG:    %[[READ_LHS_4_0:.+]] = vector.extract_strided_slice %[[READ_LHS_4]] {offsets = [0, 0, 0]
//  CHECK-DAG:    %[[READ_LHS_4_1:.+]] = vector.extract_strided_slice %[[READ_LHS_4]] {offsets = [0, 0, 1]
//  CHECK-DAG:    %[[READ_LHS_4_2:.+]] = vector.extract_strided_slice %[[READ_LHS_4]] {offsets = [0, 0, 2]
//  CHECK-DAG:    %[[READ_LHS_4_3:.+]] = vector.extract_strided_slice %[[READ_LHS_4]] {offsets = [0, 0, 3]
//  CHECK-DAG:    %[[READ_LHS_5_0:.+]] = vector.extract_strided_slice %[[READ_LHS_5]] {offsets = [0, 0, 0]
//  CHECK-DAG:    %[[READ_LHS_5_1:.+]] = vector.extract_strided_slice %[[READ_LHS_5]] {offsets = [0, 0, 1]
//  CHECK-DAG:    %[[READ_LHS_5_2:.+]] = vector.extract_strided_slice %[[READ_LHS_5]] {offsets = [0, 0, 2]
//  CHECK-DAG:    %[[READ_LHS_5_3:.+]] = vector.extract_strided_slice %[[READ_LHS_5]] {offsets = [0, 0, 3]
//  CHECK-DAG:    %[[READ_LHS_6_0:.+]] = vector.extract_strided_slice %[[READ_LHS_6]] {offsets = [0, 0, 0]
//  CHECK-DAG:    %[[READ_LHS_6_1:.+]] = vector.extract_strided_slice %[[READ_LHS_6]] {offsets = [0, 0, 1]
//  CHECK-DAG:    %[[READ_LHS_6_2:.+]] = vector.extract_strided_slice %[[READ_LHS_6]] {offsets = [0, 0, 2]
//  CHECK-DAG:    %[[READ_LHS_6_3:.+]] = vector.extract_strided_slice %[[READ_LHS_6]] {offsets = [0, 0, 3]
//  CHECK-DAG:    %[[READ_LHS_7_0:.+]] = vector.extract_strided_slice %[[READ_LHS_7]] {offsets = [0, 0, 0]
//  CHECK-DAG:    %[[READ_LHS_7_1:.+]] = vector.extract_strided_slice %[[READ_LHS_7]] {offsets = [0, 0, 1]
//  CHECK-DAG:    %[[READ_LHS_7_2:.+]] = vector.extract_strided_slice %[[READ_LHS_7]] {offsets = [0, 0, 2]
//  CHECK-DAG:    %[[READ_LHS_7_3:.+]] = vector.extract_strided_slice %[[READ_LHS_7]] {offsets = [0, 0, 3]
// Vectorization and lowering generates a lot of transpose and shape_cast that
// are only  simplified later. We could probably improve the pass organization to
// avoid it.
//  CHECK-DAG:    %[[READ_RHS_0_T:.+]] = vector.transpose %[[READ_RHS_0]], [0, 2, 1]
//  CHECK-DAG:    %[[READ_RHS_1_T:.+]] = vector.transpose %[[READ_RHS_1]], [0, 2, 1]
//  CHECK-DAG:    %[[READ_RHS_2_T:.+]] = vector.transpose %[[READ_RHS_2]], [0, 2, 1]
//  CHECK-DAG:    %[[READ_RHS_3_T:.+]] = vector.transpose %[[READ_RHS_3]], [0, 2, 1]

//  CHECK-DAG:    %[[READ_RHS_0_T_1:.+]] = vector.shape_cast %[[READ_RHS_0_T]] : vector<1x4x1xf32> to vector<4x1xf32>
//  CHECK-DAG:    %[[READ_RHS_0_T_2:.+]] = vector.transpose %[[READ_RHS_0_T_1]], [1, 0] : vector<4x1xf32> to vector<1x4xf32>
//  CHECK-DAG:    %[[READ_RHS_0_T_3:.+]] = vector.shape_cast %[[READ_RHS_0_T_2]] : vector<1x4xf32> to vector<4xf32>
//  CHECK-DAG:    %[[READ_LHS_0_0_E:.+]] = vector.extract %[[READ_LHS_0_0]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_0_0_S:.+]] = splat %[[READ_LHS_0_0_E]] : vector<4xf32>
//  CHECK-DAG:    %[[ACC_0_C:.+]] = vector.shape_cast %[[ACC_0]] : vector<1x1x4xf32> to vector<4xf32>
//  CHECK-DAG:    %[[FMA_0_0:.+]] = vector.fma %[[READ_LHS_0_0_S]], %[[READ_RHS_0_T_3]], %[[ACC_0_C]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_RHS_1_T_1:.+]] = vector.shape_cast %[[READ_RHS_1_T]] : vector<1x4x1xf32> to vector<4x1xf32>
//  CHECK-DAG:    %[[READ_RHS_1_T_2:.+]] = vector.transpose %[[READ_RHS_1_T_1]], [1, 0] : vector<4x1xf32> to vector<1x4xf32>
//  CHECK-DAG:    %[[READ_RHS_1_T_3:.+]] = vector.shape_cast %[[READ_RHS_1_T_2]] : vector<1x4xf32> to vector<4xf32>
//  CHECK-DAG:    %[[READ_LHS_0_1_E:.+]] = vector.extract %[[READ_LHS_0_1]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_0_1_S:.+]] = splat %[[READ_LHS_0_1_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_0_1:.+]] = vector.fma %[[READ_LHS_0_1_S]], %[[READ_RHS_1_T_3]], %[[FMA_0_0]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_RHS_2_T_1:.+]] = vector.shape_cast %[[READ_RHS_2_T]] : vector<1x4x1xf32> to vector<4x1xf32>
//  CHECK-DAG:    %[[READ_RHS_2_T_2:.+]] = vector.transpose %[[READ_RHS_2_T_1]], [1, 0] : vector<4x1xf32> to vector<1x4xf32>
//  CHECK-DAG:    %[[READ_RHS_2_T_3:.+]] = vector.shape_cast %[[READ_RHS_2_T_2]] : vector<1x4xf32> to vector<4xf32>
//  CHECK-DAG:    %[[READ_LHS_0_2_E:.+]] = vector.extract %[[READ_LHS_0_2]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_0_2_S:.+]] = splat %[[READ_LHS_0_2_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_0_2:.+]] = vector.fma %[[READ_LHS_0_2_S]], %[[READ_RHS_2_T_3]], %[[FMA_0_1]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_RHS_3_T_1:.+]] = vector.shape_cast %[[READ_RHS_3_T]] : vector<1x4x1xf32> to vector<4x1xf32>
//  CHECK-DAG:    %[[READ_RHS_3_T_2:.+]] = vector.transpose %[[READ_RHS_3_T_1]], [1, 0] : vector<4x1xf32> to vector<1x4xf32>
//  CHECK-DAG:    %[[READ_RHS_3_T_3:.+]] = vector.shape_cast %[[READ_RHS_3_T_2]] : vector<1x4xf32> to vector<4xf32>
//  CHECK-DAG:    %[[READ_LHS_0_3_E:.+]] = vector.extract %[[READ_LHS_0_3]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_0_3_S:.+]] = splat %[[READ_LHS_0_3_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_0_3:.+]] = vector.fma %[[READ_LHS_0_3_S]], %[[READ_RHS_3_T_3]], %[[FMA_0_2]] : vector<4xf32>

//      CHECK:    %[[FMA_0_3_C:.+]] = vector.shape_cast %[[FMA_0_3]] : vector<4xf32> to vector<1x1x4xf32>

//  CHECK-DAG:    %[[READ_LHS_1_0_E:.+]] = vector.extract %[[READ_LHS_1_0]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_1_0_S:.+]] = splat %[[READ_LHS_1_0_E]] : vector<4xf32>
//  CHECK-DAG:    %[[ACC_1_C:.+]] = vector.shape_cast %[[ACC_1]] : vector<1x1x4xf32> to vector<4xf32>
//  CHECK-DAG:    %[[FMA_1_0:.+]] = vector.fma %[[READ_LHS_1_0_S]], %[[READ_RHS_0_T_3]], %[[ACC_1_C]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_1_1_E:.+]] = vector.extract %[[READ_LHS_1_1]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_1_1_S:.+]] = splat %[[READ_LHS_1_1_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_1_1:.+]] = vector.fma %[[READ_LHS_1_1_S]], %[[READ_RHS_1_T_3]], %[[FMA_1_0]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_1_2_E:.+]] = vector.extract %[[READ_LHS_1_2]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_1_2_S:.+]] = splat %[[READ_LHS_1_2_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_1_2:.+]] = vector.fma %[[READ_LHS_1_2_S]], %[[READ_RHS_2_T_3]], %[[FMA_1_1]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_1_3_E:.+]] = vector.extract %[[READ_LHS_1_3]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_1_3_S:.+]] = splat %[[READ_LHS_1_3_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_1_3:.+]] = vector.fma %[[READ_LHS_1_3_S]], %[[READ_RHS_3_T_3]], %[[FMA_1_2]] : vector<4xf32>

//      CHECK:    %[[FMA_1_3_C:.+]] = vector.shape_cast %[[FMA_1_3]] : vector<4xf32> to vector<1x1x4xf32>

//  CHECK-DAG:    %[[READ_LHS_2_0_E:.+]] = vector.extract %[[READ_LHS_2_0]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_2_0_S:.+]] = splat %[[READ_LHS_2_0_E]] : vector<4xf32>
//  CHECK-DAG:    %[[ACC_2_C:.+]] = vector.shape_cast %[[ACC_2]] : vector<1x1x4xf32> to vector<4xf32>
//  CHECK-DAG:    %[[FMA_2_0:.+]] = vector.fma %[[READ_LHS_2_0_S]], %[[READ_RHS_0_T_3]], %[[ACC_2_C]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_2_1_E:.+]] = vector.extract %[[READ_LHS_2_1]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_2_1_S:.+]] = splat %[[READ_LHS_2_1_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_2_1:.+]] = vector.fma %[[READ_LHS_2_1_S]], %[[READ_RHS_1_T_3]], %[[FMA_2_0]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_2_2_E:.+]] = vector.extract %[[READ_LHS_2_2]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_2_2_S:.+]] = splat %[[READ_LHS_2_2_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_2_2:.+]] = vector.fma %[[READ_LHS_2_2_S]], %[[READ_RHS_2_T_3]], %[[FMA_2_1]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_2_3_E:.+]] = vector.extract %[[READ_LHS_2_3]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_2_3_S:.+]] = splat %[[READ_LHS_2_3_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_2_3:.+]] = vector.fma %[[READ_LHS_2_3_S]], %[[READ_RHS_3_T_3]], %[[FMA_2_2]] : vector<4xf32>

//      CHECK:    %[[FMA_2_3_C:.+]] = vector.shape_cast %[[FMA_2_3]] : vector<4xf32> to vector<1x1x4xf32>

//  CHECK-DAG:    %[[READ_LHS_3_0_E:.+]] = vector.extract %[[READ_LHS_3_0]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_3_0_S:.+]] = splat %[[READ_LHS_3_0_E]] : vector<4xf32>
//  CHECK-DAG:    %[[ACC_3_C:.+]] = vector.shape_cast %[[ACC_3]] : vector<1x1x4xf32> to vector<4xf32>
//  CHECK-DAG:    %[[FMA_3_0:.+]] = vector.fma %[[READ_LHS_3_0_S]], %[[READ_RHS_0_T_3]], %[[ACC_3_C]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_3_1_E:.+]] = vector.extract %[[READ_LHS_3_1]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_3_1_S:.+]] = splat %[[READ_LHS_3_1_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_3_1:.+]] = vector.fma %[[READ_LHS_3_1_S]], %[[READ_RHS_1_T_3]], %[[FMA_3_0]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_3_2_E:.+]] = vector.extract %[[READ_LHS_3_2]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_3_2_S:.+]] = splat %[[READ_LHS_3_2_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_3_2:.+]] = vector.fma %[[READ_LHS_3_2_S]], %[[READ_RHS_2_T_3]], %[[FMA_3_1]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_3_3_E:.+]] = vector.extract %[[READ_LHS_3_3]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_3_3_S:.+]] = splat %[[READ_LHS_3_3_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_3_3:.+]] = vector.fma %[[READ_LHS_3_3_S]], %[[READ_RHS_3_T_3]], %[[FMA_3_2]] : vector<4xf32>

//      CHECK:    %[[FMA_3_3_C:.+]] = vector.shape_cast %[[FMA_3_3]] : vector<4xf32> to vector<1x1x4xf32>

//  CHECK-DAG:    %[[READ_LHS_4_0_E:.+]] = vector.extract %[[READ_LHS_4_0]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_4_0_S:.+]] = splat %[[READ_LHS_4_0_E]] : vector<4xf32>
//  CHECK-DAG:    %[[ACC_4_C:.+]] = vector.shape_cast %[[ACC_4]] : vector<1x1x4xf32> to vector<4xf32>
//  CHECK-DAG:    %[[FMA_4_0:.+]] = vector.fma %[[READ_LHS_4_0_S]], %[[READ_RHS_0_T_3]], %[[ACC_4_C]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_4_1_E:.+]] = vector.extract %[[READ_LHS_4_1]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_4_1_S:.+]] = splat %[[READ_LHS_4_1_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_4_1:.+]] = vector.fma %[[READ_LHS_4_1_S]], %[[READ_RHS_1_T_3]], %[[FMA_4_0]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_4_2_E:.+]] = vector.extract %[[READ_LHS_4_2]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_4_2_S:.+]] = splat %[[READ_LHS_4_2_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_4_2:.+]] = vector.fma %[[READ_LHS_4_2_S]], %[[READ_RHS_2_T_3]], %[[FMA_4_1]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_4_3_E:.+]] = vector.extract %[[READ_LHS_4_3]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_4_3_S:.+]] = splat %[[READ_LHS_4_3_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_4_3:.+]] = vector.fma %[[READ_LHS_4_3_S]], %[[READ_RHS_3_T_3]], %[[FMA_4_2]] : vector<4xf32>

//      CHECK:    %[[FMA_4_3_C:.+]] = vector.shape_cast %[[FMA_4_3]] : vector<4xf32> to vector<1x1x4xf32>

//  CHECK-DAG:    %[[READ_LHS_5_0_E:.+]] = vector.extract %[[READ_LHS_5_0]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_5_0_S:.+]] = splat %[[READ_LHS_5_0_E]] : vector<4xf32>
//  CHECK-DAG:    %[[ACC_5_C:.+]] = vector.shape_cast %[[ACC_5]] : vector<1x1x4xf32> to vector<4xf32>
//  CHECK-DAG:    %[[FMA_5_0:.+]] = vector.fma %[[READ_LHS_5_0_S]], %[[READ_RHS_0_T_3]], %[[ACC_5_C]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_5_1_E:.+]] = vector.extract %[[READ_LHS_5_1]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_5_1_S:.+]] = splat %[[READ_LHS_5_1_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_5_1:.+]] = vector.fma %[[READ_LHS_5_1_S]], %[[READ_RHS_1_T_3]], %[[FMA_5_0]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_5_2_E:.+]] = vector.extract %[[READ_LHS_5_2]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_5_2_S:.+]] = splat %[[READ_LHS_5_2_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_5_2:.+]] = vector.fma %[[READ_LHS_5_2_S]], %[[READ_RHS_2_T_3]], %[[FMA_5_1]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_5_3_E:.+]] = vector.extract %[[READ_LHS_5_3]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_5_3_S:.+]] = splat %[[READ_LHS_5_3_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_5_3:.+]] = vector.fma %[[READ_LHS_5_3_S]], %[[READ_RHS_3_T_3]], %[[FMA_5_2]] : vector<4xf32>

//      CHECK:    %[[FMA_5_3_C:.+]] = vector.shape_cast %[[FMA_5_3]] : vector<4xf32> to vector<1x1x4xf32>

//  CHECK-DAG:    %[[READ_LHS_6_0_E:.+]] = vector.extract %[[READ_LHS_6_0]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_6_0_S:.+]] = splat %[[READ_LHS_6_0_E]] : vector<4xf32>
//  CHECK-DAG:    %[[ACC_6_C:.+]] = vector.shape_cast %[[ACC_6]] : vector<1x1x4xf32> to vector<4xf32>
//  CHECK-DAG:    %[[FMA_6_0:.+]] = vector.fma %[[READ_LHS_6_0_S]], %[[READ_RHS_0_T_3]], %[[ACC_6_C]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_6_1_E:.+]] = vector.extract %[[READ_LHS_6_1]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_6_1_S:.+]] = splat %[[READ_LHS_6_1_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_6_1:.+]] = vector.fma %[[READ_LHS_6_1_S]], %[[READ_RHS_1_T_3]], %[[FMA_6_0]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_6_2_E:.+]] = vector.extract %[[READ_LHS_6_2]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_6_2_S:.+]] = splat %[[READ_LHS_6_2_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_6_2:.+]] = vector.fma %[[READ_LHS_6_2_S]], %[[READ_RHS_2_T_3]], %[[FMA_6_1]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_6_3_E:.+]] = vector.extract %[[READ_LHS_6_3]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_6_3_S:.+]] = splat %[[READ_LHS_6_3_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_6_3:.+]] = vector.fma %[[READ_LHS_6_3_S]], %[[READ_RHS_3_T_3]], %[[FMA_6_2]] : vector<4xf32>

//      CHECK:    %[[FMA_6_3_C:.+]] = vector.shape_cast %[[FMA_6_3]] : vector<4xf32> to vector<1x1x4xf32>

//  CHECK-DAG:    %[[READ_LHS_7_0_E:.+]] = vector.extract %[[READ_LHS_7_0]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_7_0_S:.+]] = splat %[[READ_LHS_7_0_E]] : vector<4xf32>
//  CHECK-DAG:    %[[ACC_7_C:.+]] = vector.shape_cast %[[ACC_7]] : vector<1x1x4xf32> to vector<4xf32>
//  CHECK-DAG:    %[[FMA_7_0:.+]] = vector.fma %[[READ_LHS_7_0_S]], %[[READ_RHS_0_T_3]], %[[ACC_7_C]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_7_1_E:.+]] = vector.extract %[[READ_LHS_7_1]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_7_1_S:.+]] = splat %[[READ_LHS_7_1_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_7_1:.+]] = vector.fma %[[READ_LHS_7_1_S]], %[[READ_RHS_1_T_3]], %[[FMA_7_0]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_7_2_E:.+]] = vector.extract %[[READ_LHS_7_2]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_7_2_S:.+]] = splat %[[READ_LHS_7_2_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_7_2:.+]] = vector.fma %[[READ_LHS_7_2_S]], %[[READ_RHS_2_T_3]], %[[FMA_7_1]] : vector<4xf32>

//  CHECK-DAG:    %[[READ_LHS_7_3_E:.+]] = vector.extract %[[READ_LHS_7_3]][0, 0, 0] : vector<1x1x1xf32>
//  CHECK-DAG:    %[[READ_LHS_7_3_S:.+]] = splat %[[READ_LHS_7_3_E]] : vector<4xf32>
//  CHECK-DAG:    %[[FMA_7_3:.+]] = vector.fma %[[READ_LHS_7_3_S]], %[[READ_RHS_3_T_3]], %[[FMA_7_2]] : vector<4xf32>

//      CHECK:    %[[FMA_7_3_C:.+]] = vector.shape_cast %[[FMA_7_3]] : vector<4xf32> to vector<1x1x4xf32>

//      CHECK:  scf.yield %[[FMA_0_3_C]], %[[FMA_1_3_C]], %[[FMA_2_3_C]],
// CHECK-SAME:  %[[FMA_3_3_C]], %[[FMA_4_3_C]], %[[FMA_5_3_C]], %[[FMA_6_3_C]],
// CHECK-SAME:  %[[FMA_7_3_C]]

//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#0, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C0]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#1, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C1]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#2, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C2]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#3, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C3]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#4, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C4]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#5, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C5]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#6, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C6]], %[[C0]]]
//  CHECK-DAG:  vector.transfer_write %[[FOR_RES]]#7, %[[SUBVIEW_RESULT_2]][%[[C0]], %[[C7]], %[[C0]]]

// -----

hal.executable @fused_fill_batch_matmul attributes {sym_visibility = "private"} {
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan_spirv, filter="vulkan*" {
    hal.executable.entry_point @fused_fill_batch_matmul attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, ARM:IntegratedGPU, {}>}  {
      func @fused_fill_batch_matmul() {
        %zero = constant 0.0 : f32
        %c0 = constant 0 : index
        %c4 = constant 4 : index
        %c1024 = constant 1024 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<4x1024x1024xf32>
        %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : memref<4x1024x1024xf32>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : memref<4x1024x1024xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_z, %workgroup_size_z]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_z, %workgroup_size_z]
        scf.for %arg0 = %3 to %c4 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c1024 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c1024 step %8 {
              %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4)>(%arg0)[%workgroup_size_z]
              %10 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1024)>(%arg1)[%workgroup_size_y]
              %11 = memref.subview %0[%arg0, %arg1, 0] [%9, %10, 1024] [1, 1, 1] : memref<4x1024x1024xf32> to memref<?x?x1024xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
              %12 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1024)>(%arg2)[%workgroup_size_x]
              %13 = memref.subview %1[%arg0, 0, %arg2] [%9, 1024, %12] [1, 1, 1] : memref<4x1024x1024xf32> to memref<?x1024x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
              %14 = memref.subview %2[%arg0, %arg1, %arg2] [%9, %10, %12] [1, 1, 1] : memref<4x1024x1024xf32> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
              linalg.fill(%zero, %14) : f32, memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
              linalg.batch_matmul {__internal_linalg_transform__ = "workgroup"} ins(%11, %13 : memref<?x?x1024xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>, memref<?x1024x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>) outs(%14 : memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>)
            }
          }
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

//    CHECK-LABEL: func @fused_fill_batch_matmul
//  CHECK-COUNT-8:   vector.transfer_write
//  CHECK-COUNT-8:   vector.transfer_read
//          CHECK:   %[[FOR_RES:.+]]:8 = scf.for
// CHECK-COUNT-12:     vector.transfer_read
// CHECK-COUNT-32:     vector.fma
//      CHECK:         scf.yield
//  CHECK-COUNT-8:    vector.transfer_write %[[FOR_RES]]
//          CHECK:    return
