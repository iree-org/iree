// RUN: iree-opt -split-input-file -iree-codegen-linalg-tile-and-fuse=use-vectorization %s | IreeFileCheck %s

module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
      [Shader, CooperativeMatrixNV],
      [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix]>,
      {max_compute_workgroup_invocations = 512 : i32,
       max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @matmul_static_shape()
    attributes {vkspv.num_workgroups_fn = @matmul_static_shape__num_workgroups__} {
    %arg0 = iree.placeholder for "interface buffer"
      {binding = @legacy_io::@arg0, operand_result_num = 0 : i32} : memref<128x64xf16>
    %arg1 = iree.placeholder for "interface buffer"
      {binding = @legacy_io::@arg1, operand_result_num = 1 : i32} : memref<64x256xf16>
    %ret0 = iree.placeholder for "interface buffer"
      {binding = @legacy_io::@ret0, operand_result_num = 2 : i32} : memref<128x256xf16>
    linalg.matmul ins(%arg0, %arg1 : memref<128x64xf16>, memref<64x256xf16>)
                 outs(%ret0 : memref<128x256xf16>)
    return
  }
  func @matmul_static_shape__num_workgroups__
    (!shapex.ranked_shape<[128, 64]>, !shapex.ranked_shape<[64, 256]>,
     !shapex.ranked_shape<[128, 256]>) -> (index, index, index)
    attributes {sym_visibility = "private"}
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 32)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>
//      CHECK: func @matmul_static_shape
//  CHECK-DAG:  %[[ARG0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg0
//  CHECK-DAG:  %[[ARG1:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg1
//  CHECK-DAG:  %[[RET0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@ret0
//  CHECK-DAG:  %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:  %[[CST:.+]] = constant 0.0
//  CHECK-DAG:  %[[C4:.+]] = constant 4 : index
//      CHECK:  %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//      CHECK:  %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//      CHECK:  %[[BOFFSET_Y:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//      CHECK:  %[[SUBVIEW_LHS:.+]] = subview %[[ARG0]]
// CHECK-SAME:    [%[[BOFFSET_Y]], 0] [32, 64]
//      CHECK:  %[[BOFFSET_X:.+]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//      CHECK:  %[[SUBVIEW_RHS:.+]] = subview %[[ARG1]]
// CHECK-SAME:    [0, %[[BOFFSET_X]]] [64, 32]
//      CHECK:  %[[BOFFSET_Y_2:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//      CHECK:  %[[BOFFSET_X_2:.+]] = affine.apply #[[MAP0]]()[%[[BIDX]]]
//      CHECK:  %[[SUBVIEW_RESULT:.+]] = subview %[[RET0]]
// CHECK-SAME:    [%[[BOFFSET_Y_2]], %[[BOFFSET_X_2]]] [32, 32]
//      CHECK:  %[[SGID:.+]] = gpu.subgroup_id
//      CHECK:  %[[SGID_Y:.+]] = divi_signed %[[SGID]], %[[C4]]
//      CHECK:  scf.for %[[IV2:.+]] =
//      CHECK:    %[[SGOFFSET_Y:.+]] = affine.apply #[[MAP3]]()[%[[SGID_Y]]]
//      CHECK:    %[[SUBVIEW2_LHS:.+]] = subview %[[SUBVIEW_LHS]]
// CHECK-SAME:      [%[[SGOFFSET_Y]], %[[IV2]]] [8, 16]
//      CHECK:    %[[SGOFFSET_X:.+]] = affine.apply #[[MAP3]]()[%[[SGID]]]
//      CHECK:    %[[SUBVIEW2_RHS:.+]] = subview %[[SUBVIEW_RHS]]
// CHECK-SAME:      [%[[IV2]], %[[SGOFFSET_X]]] [16, 8]
//      CHECK:    %[[SGOFFSET_Y_2:.+]] = affine.apply #[[MAP3]]()[%[[SGID_Y]]]
//      CHECK:    %[[SGOFFSET_X_2:.+]] = affine.apply #[[MAP3]]()[%[[SGID]]]
//      CHECK:    %[[SUBVIEW2_RESULT:.+]] = subview %[[SUBVIEW_RESULT]]
// CHECK-SAME:      [%[[SGOFFSET_Y_2]], %[[SGOFFSET_X_2]]] [8, 8]
//      CHECK:    %[[VTR_LHS:.+]] = vector.transfer_read %[[SUBVIEW2_LHS]]
// CHECK-SAME:      [%[[C0]], %[[C0]]], %[[CST]] {masked = [false, false]}
//      CHECK:    %[[VTR_RHS:.+]] = vector.transfer_read %[[SUBVIEW2_RHS]]
// CHECK-SAME:      [%[[C0]], %[[C0]]], %[[CST]] {masked = [false, false]}
//      CHECK:    %[[VTR_RESULT:.+]] = vector.transfer_read %[[SUBVIEW2_RESULT]]
// CHECK-SAME:      [%[[C0]], %[[C0]]], %[[CST]] {masked = [false, false]}
//      CHECK:    %[[VECTOR_CONTRACT:.+]] = vector.contract
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:      vector<8x16xf16>, vector<16x8xf16> into vector<8x8xf16>
//      CHECK:    vector.transfer_write %[[VECTOR_CONTRACT]], %[[SUBVIEW2_RESULT]]
// CHECK-SAME:      masked = [false, false]
