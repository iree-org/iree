// RUN: iree-opt -split-input-file -iree-codegen-linalg-tile-and-fuse %s | IreeFileCheck %s

// Test to check that convolution with padding is not tiled.
module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
    [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @conv_padding() {
    %0 = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg0} : memref<?x?x?x?xf32>
    %1 = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg1} : memref<?x?x?x?xf32>
    %2 = iree.placeholder for "interace buffer" {binding = @legacy_io::@ret0} : memref<?x?x?x?xf32>
    linalg.conv(%0, %1, %2)
      {dilations = [1, 1],
       padding = dense<[[1, 1], [0, 1]]> : tensor<2x2xi64>, strides = [1, 1]} :
      memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
//       CHECK: func @conv_padding()
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg0}
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg1}
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder for "interace buffer" {binding = @legacy_io::@ret0}
//       CHECK:   linalg.conv
//  CHECK-SAME:     %[[ARG0]]
//  CHECK-SAME:     %[[ARG1]]
//  CHECK-SAME:     %[[RET0]]

// -----

module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
    [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @conv_no_padding()
    attributes {hal.num_workgroups_fn = @conv_no_padding__num_workgroups__} {
    %0 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@arg0, operand_result_index = 0 : i32} : memref<?x?x?x?xf32>
    %1 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@arg1, operand_result_index = 1 : i32} : memref<?x?x?x?xf32>
    %2 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@ret0, operand_result_index = 2 : i32} : memref<?x?x?x?xf32>
    linalg.conv(%0, %1, %2) {dilations = [1, 1], strides = [1, 1]} :
      memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
    return
  }
  func @conv_no_padding__num_workgroups__
    (!shapex.ranked_shape<[?,?,?,?]>, !shapex.ranked_shape<[?,?,?,?]>,
     !shapex.ranked_shape<[?,?,?,?]>) -> (index, index, index)
    attributes {sym_visibility = "private"}
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 4)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 32)>
//       CHECK: func @conv_no_padding()
//  CHECK-SAME:   hal.num_workgroups_fn = @[[NUM_WORKGROUPS_FN:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   local_size = dense<[32, 4, 1]>
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@ret0
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-DAG:   %[[BIDZ:.+]] = "gpu.block_id"() {dimension = "z"}
//       CHECK:   %[[LBY:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[LBX:.+]] = affine.apply #[[MAP1]]()[%[[BIDX]]]
//       CHECK:   %[[VIEW1:.+]] = subview %[[ARG1]]
//  CHECK-SAME:     [%[[BIDZ]], %[[LBY]], %[[LBX]], 0]
//       CHECK:   %[[LBY_2:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[LBX_2:.+]] = affine.apply #[[MAP1]]()[%[[BIDX]]]
//       CHECK:   %[[VIEW2:.+]] = subview %[[RET0]]
//  CHECK-SAME:     [%[[BIDZ]], %[[LBY_2]], %[[LBX_2]], 0]
//       CHECK:   linalg.conv
//  CHECK-SAME:     %[[ARG0]], %[[VIEW1]], %[[VIEW2]]
//  CHECK-SAME:     "workgroup"
//       CHECK: func private @[[NUM_WORKGROUPS_FN]]
//   CHECK-DAG:   %[[C0:.+]] = constant 0
//   CHECK-DAG:   %[[C1:.+]] = constant 1
//   CHECK-DAG:   %[[C2:.+]] = constant 2
//   CHECK-DAG:   %[[C32:.+]] = constant 32
//   CHECK-DAG:   %[[C31:.+]] = constant 31
//   CHECK-DAG:   %[[C4:.+]] = constant 4
//   CHECK-DAG:   %[[C3:.+]] = constant 3
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder {{.+}} {binding = @legacy_io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder {{.+}} {binding = @legacy_io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder {{.+}} {binding = @legacy_io::@ret0
//   CHECK-DAG:   %[[DIM0:.+]] = dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:   %[[DIM1:.+]] = dim %[[RET0]], %[[C1]]
//   CHECK-DAG:   %[[DIM2:.+]] = dim %[[RET0]], %[[C2]]
//       CHECK:   %[[T0:.+]] = addi %[[DIM2]], %[[C31]]
//   CHECK-DAG:   %[[NBX:.+]] = divi_signed %[[T0]], %[[C32]]
//       CHECK:   %[[T1:.+]] = addi %[[DIM1]], %[[C3]]
//   CHECK-DAG:   %[[NBY:.+]] = divi_signed %[[T1]], %[[C4]]
//       CHECK:   return %[[NBX]], %[[NBY]], %[[DIM0]]


// -----

module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
    [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @matmul() attributes {hal.num_workgroups_fn = @matmul__num_workgroups__} {
    %0 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@arg0, operand_result_index = 0 : i32} : memref<?x?xf32>
    %1 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@arg1, operand_result_index = 1 : i32} : memref<?x?xf32>
    %2 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@ret0, operand_result_index = 2 : i32} : memref<?x?xf32>
    linalg.matmul ins(%0, %1 : memref<?x?xf32>, memref<?x?xf32>)
                 outs(%2 : memref<?x?xf32>)
    return
  }
  func @matmul__num_workgroups__
    (!shapex.ranked_shape<[?,?]>, !shapex.ranked_shape<[?,?]>,
     !shapex.ranked_shape<[?,?]>) -> (index, index, index)
    attributes {sym_visibility = "private"}
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 8)>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0] -> (s0 * 16)>
//       CHECK: func @matmul()
//  CHECK-SAME:   hal.num_workgroups_fn = @[[NUM_WORKGROUPS_FN:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   local_size = dense<[16, 8, 1]>
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@ret0
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-NOT:   scf.parallel
//   CHECK-NOT:   scf.for
//       CHECK:   %[[LBY:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[VIEW0:.+]] = subview %[[ARG0]][%[[LBY]], 0]
//       CHECK:   %[[LBX:.+]] = affine.apply #[[MAP3]]()[%[[BIDX]]]
//       CHECK:   %[[VIEW1:.+]] = subview %[[ARG1]][0, %[[LBX]]]
//       CHECK:   %[[LBY_2:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[LBX_2:.+]] = affine.apply #[[MAP3]]()[%[[BIDX]]]
//       CHECK:   %[[VIEW2:.+]] = subview %[[RET0]][%[[LBY_2]], %[[LBX_2]]]
//       CHECK:   linalg.matmul
//  CHECK-SAME:     "workgroup"
//  CHECK-SAME:     ins(%[[VIEW0]], %[[VIEW1]]
//  CHECK-SAME:     outs(%[[VIEW2]]
//       CHECK: func private @[[NUM_WORKGROUPS_FN]]
//   CHECK-DAG:   %[[C8:.+]] = constant 8 : index
//   CHECK-DAG:   %[[C7:.+]] = constant 7 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[C16:.+]] = constant 16 : index
//   CHECK-DAG:   %[[C15:.+]] = constant 15 : index
//       CHECK:   %[[DIM0:.+]] = dim %{{.*}}, %[[C0]]
//       CHECK:   %[[DIM1:.+]] = dim %{{.*}}, %[[C1]]
//       CHECK:   %[[T0:.+]] = addi %[[DIM1]], %[[C15]]
//       CHECK:   %[[T1:.+]] = divi_signed %[[T0]], %[[C16]]
//       CHECK:   %[[T2:.+]] = addi %[[DIM0]], %[[C7]]
//       CHECK:   %[[T3:.+]] = divi_signed %[[T2]], %[[C8]]
//       CHECK:   return %[[T1]], %[[T3]], %[[C1]]

// -----

module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
    [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @pooling_sum_no_padding()
    attributes {hal.num_workgroups_fn = @pooling_sum_no_padding__num_workgroups__} {
    %0 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@arg0, operand_result_index = 0 : i32} : memref<?x?xf32>
    %1 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@arg1, operand_result_index = 1 : i32} : memref<?x?xf32>
    %2 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@ret0, operand_result_index = 2 : i32} : memref<?x?xf32>
    linalg.pooling_max(%0, %1, %2) {dilations = [1, 1], strides = [1, 1]} :
      memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
    return
  }
  func @pooling_sum_no_padding__num_workgroups__
    (!shapex.ranked_shape<[?,?]>, !shapex.ranked_shape<[?,?]>,
     !shapex.ranked_shape<[?,?]>) -> (index, index, index)
    attributes {sym_visibility = "private"}
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 4)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 32)>
//       CHECK: func @pooling_sum_no_padding()
//  CHECK-SAME:   hal.num_workgroups_fn = @[[NUM_WORKGROUPS_FN:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   local_size = dense<[32, 4, 1]>
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@ret0
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//       CHECK:   %[[LBY:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[LBX:.+]] = affine.apply #[[MAP1]]()[%[[BIDX]]]
//       CHECK:   %[[VIEW0:.+]] = subview %[[ARG0]][%[[LBY]], %[[LBX]]]
//       CHECK:   %[[LBY2:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[LBX2:.+]] = affine.apply #[[MAP1]]()[%[[BIDX]]]
//       CHECK:   %[[VIEW2:.+]] = subview %[[RET0]][%[[LBY2]], %[[LBX2]]]
//       CHECK:   linalg.pooling_max
//  CHECK-SAME:     %[[VIEW0]], %[[ARG1]], %[[VIEW2]]
//  CHECK-SAME:     "workgroup"
//       CHECK: func private @[[NUM_WORKGROUPS_FN]]
//   CHECK-DAG:   %[[C0:.+]] = constant 0
//   CHECK-DAG:   %[[C1:.+]] = constant 1
//   CHECK-DAG:   %[[C32:.+]] = constant 32
//   CHECK-DAG:   %[[C31:.+]] = constant 31
//   CHECK-DAG:   %[[C4:.+]] = constant 4
//   CHECK-DAG:   %[[C3:.+]] = constant 3
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder {{.+}} {binding = @legacy_io::@ret0
//   CHECK-DAG:   %[[DIM0:.+]] = dim %[[RET0]], %[[C0]]
//   CHECK-DAG:   %[[DIM1:.+]] = dim %[[RET0]], %[[C1]]
//       CHECK:   %[[T0:.+]] = addi %[[DIM1]], %[[C31]]
//   CHECK-DAG:   %[[NBX:.+]] = divi_signed %[[T0]], %[[C32]]
//       CHECK:   %[[T1:.+]] = addi %[[DIM0]], %[[C3]]
//   CHECK-DAG:   %[[NBY:.+]] = divi_signed %[[T1]], %[[C4]]
//       CHECK:   return %[[NBX]], %[[NBY]], %[[C1]]

// -----

module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
    [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @pooling_max_4D()
    attributes {hal.num_workgroups_fn = @pooling_max_4D__num_workgroups__} {
    %0 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@arg0, operand_result_index = 0 : i32} : memref<?x?x?x?xf32>
    %1 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@arg1, operand_result_index = 1 : i32} : memref<?x?x?x?xf32>
    %2 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@ret0, operand_result_index = 2 : i32} : memref<?x?x?x?xf32>
    linalg.pooling_max(%0, %1, %2) {dilations = [1, 1, 1, 1], strides = [1, 1, 1, 1]} :
      memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
    return
  }
  func @pooling_max_4D__num_workgroups__
    (!shapex.ranked_shape<[?,?,?,?]>, !shapex.ranked_shape<[?,?,?,?]>,
     !shapex.ranked_shape<[?,?,?,?]>) -> (index, index, index)
    attributes {sym_visibility = "private"}
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 4)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 32)>
//       CHECK: func @pooling_max_4D()
//  CHECK-SAME:   hal.num_workgroups_fn = @[[NUM_WORKGROUPS_FN:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   local_size = dense<[32, 4, 1]>
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@ret0
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//       CHECK:   %[[LBY:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[LBX:.+]] = affine.apply #[[MAP2]]()[%[[BIDX]]]
//       CHECK:   %[[VIEW0:.+]] = subview %[[ARG0]][0, %[[LBY]], %[[LBX]], 0]
//       CHECK:   %[[LBY2:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[LBX2:.+]] = affine.apply #[[MAP2]]()[%[[BIDX]]]
//       CHECK:   %[[VIEW2:.+]] = subview %[[RET0]][0, %[[LBY2]], %[[LBX2]], 0]
//       CHECK:   linalg.pooling_max
//  CHECK-SAME:     %[[VIEW0]], %[[ARG1]], %[[VIEW2]]
//  CHECK-SAME:     "workgroup"
//       CHECK: func private @[[NUM_WORKGROUPS_FN]]
//   CHECK-DAG:   %[[C1:.+]] = constant 1
//   CHECK-DAG:   %[[C32:.+]] = constant 32
//   CHECK-DAG:   %[[C31:.+]] = constant 31
//   CHECK-DAG:   %[[C4:.+]] = constant 4
//   CHECK-DAG:   %[[C3:.+]] = constant 3
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder {{.+}} {binding = @legacy_io::@ret0
//   CHECK-DAG:   %[[DIM0:.+]] = dim %[[RET0]], %[[C1]]
//   CHECK-DAG:   %[[DIM1:.+]] = dim %[[RET0]], %[[C2]]
//       CHECK:   %[[T0:.+]] = addi %[[DIM1]], %[[C31]]
//   CHECK-DAG:   %[[NBX:.+]] = divi_signed %[[T0]], %[[C32]]
//       CHECK:   %[[T1:.+]] = addi %[[DIM0]], %[[C3]]
//   CHECK-DAG:   %[[NBY:.+]] = divi_signed %[[T1]], %[[C4]]
//       CHECK:   return %[[NBX]], %[[NBY]], %[[C1]]

// -----

module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
    [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @matmul_fusion() attributes {hal.num_workgroups_fn = @matmul_fusion__num_workgroups__} {
    %0 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@arg0, operand_result_index = 0 : i32} : memref<?x?xf32>
    %1 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@arg1, operand_result_index = 1 : i32} : memref<?x?xf32>
    %2 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@ret0, operand_result_index = 2 : i32} : memref<?x?xf32>
    %cst = constant 0.000000e+00 : f32
    linalg.fill(%2, %cst) : memref<?x?xf32>, f32
    linalg.matmul ins(%0, %1 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%2 : memref<?x?xf32>)
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

//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 8)>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0] -> (s0 * 16)>
//       CHECK: func @matmul_fusion()
//  CHECK-SAME:   local_size = dense<[16, 8, 1]>
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@ret0
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-NOT:   scf.parallel
//   CHECK-NOT:   scf.for
//       CHECK:   %[[LBY:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[VIEW0:.+]] = subview %[[ARG0]][%[[LBY]], 0]
//       CHECK:   %[[LBX:.+]] = affine.apply #[[MAP3]]()[%[[BIDX]]]
//       CHECK:   %[[VIEW1:.+]] = subview %[[ARG1]][0, %[[LBX]]]
//       CHECK:   %[[LBY_2:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[LBX_2:.+]] = affine.apply #[[MAP3]]()[%[[BIDX]]]
//       CHECK:   %[[VIEW2:.+]] = subview %[[RET0]][%[[LBY_2]], %[[LBX_2]]]
//       CHECK:   %[[VIEW3:.+]] = subview %[[RET0]][%[[LBY]], %[[LBX]]]
//       CHECK:   linalg.fill(%[[VIEW3]], %{{.+}})
//  CHECK-SAME:     "workgroup"
//       CHECK:   linalg.matmul
//  CHECK-SAME:     "workgroup"
//  CHECK-SAME:     ins(%[[VIEW0]], %[[VIEW1]]
//  CHECK-SAME:     outs(%[[VIEW2]]

// -----

module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
    [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @conv_no_padding_fusion() {
    %0 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@arg0, operand_result_index = 0 : i32} : memref<?x?x?x?xf32>
    %1 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@arg1, operand_result_index = 1 : i32} : memref<?x?x?x?xf32>
    %2 = iree.placeholder for "interace buffer"
      {binding = @legacy_io::@ret0, operand_result_index = 2 : i32} : memref<?x?x?x?xf32>
    %cst = constant 0.000000e+00 : f32
    linalg.fill(%2, %cst) : memref<?x?x?x?xf32>, f32
    linalg.conv(%0, %1, %2) {dilations = [1, 1], strides = [1, 1]} :
      memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 4)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 32)>
//       CHECK: func @conv_no_padding_fusion()
//  CHECK-SAME:   local_size = dense<[32, 4, 1]>
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder {{.*}} {binding = @legacy_io::@ret0
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-DAG:   %[[BIDZ:.+]] = "gpu.block_id"() {dimension = "z"}
//       CHECK:   %[[LBY:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[LBX:.+]] = affine.apply #[[MAP1]]()[%[[BIDX]]]
//       CHECK:   %[[VIEW1:.+]] = subview %[[ARG1]]
//  CHECK-SAME:     [%[[BIDZ]], %[[LBY]], %[[LBX]], 0]
//       CHECK:   %[[LBY_2:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[LBX_2:.+]] = affine.apply #[[MAP1]]()[%[[BIDX]]]
//       CHECK:   %[[VIEW2:.+]] = subview %[[RET0]]
//  CHECK-SAME:     [%[[BIDZ]], %[[LBY_2]], %[[LBX_2]], 0]
//       CHECK:   %[[VIEW3:.+]] = subview %[[RET0]]
//  CHECK-SAME:     [%[[BIDZ]], %[[LBY_2]], %[[LBX_2]], 0]
//       CHECK:   linalg.fill(%[[VIEW3]], %{{.*}})
//  CHECK-SAME:     "workgroup"
//       CHECK:   linalg.conv
//  CHECK-SAME:     %[[ARG0]], %[[VIEW1]], %[[VIEW2]]
//  CHECK-SAME:     "workgroup"
