
module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
    [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @matmul_fusion() attributes {vkspv.num_workgroups_fn = @matmul_fusion__num_workgroups__} {
    %0 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@arg0, operand_result_index = 0 : i32} : memref<?x?xf32>
    %1 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@arg1, operand_result_index = 1 : i32} : memref<?x?xf32>
    %2 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@arg2, operand_result_index = 2 : i32} : memref<?xf32>
    %3 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@ret0, operand_result_index = 3 : i32} : memref<?x?xf32>
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    %c1 = constant 0 : index
    %d0 = dim %0, %c0 : memref<?x?xf32>
    %d1 = dim %1, %c1 : memref<?x?xf32>
    %4 = alloc(%d0, %d1) : memref<?x?xf32>
    linalg.fill(%4, %cst) : memref<?x?xf32>, f32
    linalg.matmul ins(%0, %1 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%4 : memref<?x?xf32>)
    linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]}
      ins(%4, %2 : memref<?x?xf32>, memref<?xf32>)
      outs(%3 : memref<?x?xf32>) {
      ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32) :
        %5 = addf %arg0, %arg1 : f32
        linalg.yield %5 : f32
      }
    return
  }
  func @matmul_fusion__num_workgroups__
    (!shapex.ranked_shape<[?,?]>, !shapex.ranked_shape<[?,?]>,
     !shapex.ranked_shape<[?,?]>) -> (index, index, index)
    attributes {sym_visibility = "private"}
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
