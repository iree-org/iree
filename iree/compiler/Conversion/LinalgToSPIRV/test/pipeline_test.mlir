// RUN: iree-opt -split-input-file -iree-codegen-linalg-to-spirv-pipeline=use-vectorization %s | IreeFileCheck %s

module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
      [Float16, Shader, CooperativeMatrixNV],
      [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix]>,
      {max_compute_workgroup_invocations = 512 : i32,
       max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @matmul_static_shape()
    attributes {vkspv.num_workgroups_fn = @matmul_static_shape__num_workgroups__} {
    %0 = iree.placeholder for "interface buffer"
      {binding = @legacy_io::@arg0, operand_result_num = 0} : memref<128x64xf16>
    %1 = iree.placeholder for "interface buffer"
      {binding = @legacy_io::@arg1, operand_result_num = 1} : memref<64x256xf16>
    %2 = iree.placeholder for "interface buffer"
      {binding = @legacy_io::@ret0, operand_result_num = 2} : memref<128x256xf16>
    linalg.matmul %0, %1, %2 :
      (memref<128x64xf16>, memref<64x256xf16>, memref<128x256xf16>)
    return
  }
  func @matmul_static_shape__num_workgroups__
    (!shapex.ranked_shape<[128, 64]>, !shapex.ranked_shape<[64, 256]>,
     !shapex.ranked_shape<[128, 256]>) -> (index, index, index)
    attributes {sym_visibility = "private"}
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}
// CHECK-LABEL: spv.func @matmul_static_shape
//       CHECK:   spv.CooperativeMatrixLoadNV
//       CHECK:   spv.CooperativeMatrixLoadNV
//       CHECK:   spv.CooperativeMatrixLoadNV
//       CHECK:   spv.CooperativeMatrixMulAddNV
//       CHECK:   spv.CooperativeMatrixStoreNV
