// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-linalg-tile-and-fuse))" -iree-spirv-enable-vectorization -canonicalize -cse %s | IreeFileCheck %s

hal.executable @conv_no_padding attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @conv_no_padding attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x?xf32>, !flow.dispatch.input<?x?xf32>,
        !flow.dispatch.output<?x?xf32>) -> ()}
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
      func private @conv_no_padding__num_workgroups__
        (!shapex.ranked_shape<[?,?,?,?]>, !shapex.ranked_shape<[?,?,?,?]>,
         !shapex.ranked_shape<[?,?,?,?]>) -> (index, index, index)
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 4)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 32)>
//       CHECK: func @conv_no_padding()
//  CHECK-SAME:   hal.num_workgroups_fn = @[[NUM_WORKGROUPS_FN:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   local_size = dense<[32, 4, 1]>
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder {{.*}} @legacy_io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder {{.*}} @legacy_io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder {{.*}} @legacy_io::@ret0
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-DAG:   %[[BIDZ:.+]] = "gpu.block_id"() {dimension = "z"}
//       CHECK:   %[[LBY:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[LBX:.+]] = affine.apply #[[MAP2]]()[%[[BIDX]]]
//       CHECK:   %[[SV_ARG1:.+]] = subview %[[ARG1]]
//  CHECK-SAME:     [%[[BIDZ]], %[[LBY]], %[[LBX]], 0]
//       CHECK:   %[[SV_RET0:.+]] = subview %[[RET0]]
//  CHECK-SAME:     [%[[BIDZ]], %[[LBY]], %[[LBX]], 0]
//       CHECK:   linalg.conv
//  CHECK-SAME:     %[[ARG0]], %[[SV_ARG1]], %[[SV_RET0]]
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
//       CHECK:   %[[T1:.+]] = addi %[[DIM1]], %[[C3]]
//   CHECK-DAG:   %[[NBY:.+]] = divi_signed %[[T1]], %[[C4]]
//       CHECK:   %[[T0:.+]] = addi %[[DIM2]], %[[C31]]
//   CHECK-DAG:   %[[NBX:.+]] = divi_signed %[[T0]], %[[C32]]
//       CHECK:   return %[[NBX]], %[[NBY]], %[[DIM0]]


// -----

hal.executable @matmul attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @matmul attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x?xf32>, !flow.dispatch.input<?x?xf32>,
        !flow.dispatch.output<?x?xf32>) -> ()}
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
      func private @matmul__num_workgroups__
        (!shapex.ranked_shape<[?,?]>, !shapex.ranked_shape<[?,?]>,
         !shapex.ranked_shape<[?,?]>) -> (index, index, index)
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 8)>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0] -> (s0 * 16)>
//       CHECK: func @matmul()
//  CHECK-SAME:   hal.num_workgroups_fn = @[[NUM_WORKGROUPS_FN:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   local_size = dense<[16, 8, 1]>
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder {{.*}} @legacy_io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder {{.*}} @legacy_io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder {{.*}} @legacy_io::@ret0
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-NOT:   scf.parallel
//   CHECK-NOT:   scf.for
//       CHECK:   %[[LBY:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[SV_ARG0:.+]] = subview %[[ARG0]][%[[LBY]], 0]
//       CHECK:   %[[LBX:.+]] = affine.apply #[[MAP3]]()[%[[BIDX]]]
//       CHECK:   %[[SV_ARG1:.+]] = subview %[[ARG1]][0, %[[LBX]]]
//       CHECK:   %[[SV_RET0:.+]] = subview %[[RET0]][%[[LBY]], %[[LBX]]]
//       CHECK:   linalg.matmul
//  CHECK-SAME:     "workgroup"
//  CHECK-SAME:     ins(%[[SV_ARG0]], %[[SV_ARG1]]
//  CHECK-SAME:       )
//  CHECK-SAME:     outs(%[[SV_RET0]]
//  CHECK-SAME:       )
//       CHECK: func private @[[NUM_WORKGROUPS_FN]]
//   CHECK-DAG:   %[[C8:.+]] = constant 8 : index
//   CHECK-DAG:   %[[C7:.+]] = constant 7 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[C16:.+]] = constant 16 : index
//   CHECK-DAG:   %[[C15:.+]] = constant 15 : index
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder {{.+}} {binding = @legacy_io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder {{.+}} {binding = @legacy_io::@arg1
//       CHECK:   %[[DIM0:.+]] = dim %[[ARG0]], %[[C0]]
//       CHECK:   %[[DIM1:.+]] = dim %[[ARG1]], %[[C1]]
//       CHECK:   %[[T2:.+]] = addi %[[DIM0]], %[[C7]]
//       CHECK:   %[[T3:.+]] = divi_signed %[[T2]], %[[C8]]
//       CHECK:   %[[T0:.+]] = addi %[[DIM1]], %[[C15]]
//       CHECK:   %[[T1:.+]] = divi_signed %[[T0]], %[[C16]]
//       CHECK:   return %[[T1]], %[[T3]], %[[C1]]

// -----

hal.executable @pooling_sum_no_padding attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @pooling_sum_no_padding attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x?xf32>, !flow.dispatch.input<?x?xf32>,
        !flow.dispatch.output<?x?xf32>) -> ()}
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
      func private @pooling_sum_no_padding__num_workgroups__
        (!shapex.ranked_shape<[?,?]>, !shapex.ranked_shape<[?,?]>,
         !shapex.ranked_shape<[?,?]>) -> (index, index, index)
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
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
//       CHECK:   %[[SV_ARG0:.+]] = subview %[[ARG0]][%[[LBY]], %[[LBX]]]
//       CHECK:   %[[SV_RET0:.+]] = subview %[[RET0]][%[[LBY]], %[[LBX]]]
//       CHECK:   linalg.pooling_max
//  CHECK-SAME:     %[[SV_ARG0]], %[[ARG1]], %[[SV_RET0]]
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
//       CHECK:   %[[T1:.+]] = addi %[[DIM0]], %[[C3]]
//   CHECK-DAG:   %[[NBY:.+]] = divi_signed %[[T1]], %[[C4]]
//       CHECK:   %[[T0:.+]] = addi %[[DIM1]], %[[C31]]
//   CHECK-DAG:   %[[NBX:.+]] = divi_signed %[[T0]], %[[C32]]
//       CHECK:   return %[[NBX]], %[[NBY]], %[[C1]]

// -----

hal.executable @pooling_max_4D attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @pooling_max_4D attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x?xf32>, !flow.dispatch.input<?x?xf32>,
        !flow.dispatch.output<?x?xf32>) -> ()}
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
      func private @pooling_max_4D__num_workgroups__
        (!shapex.ranked_shape<[?,?,?,?]>, !shapex.ranked_shape<[?,?,?,?]>,
         !shapex.ranked_shape<[?,?,?,?]>) -> (index, index, index)
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
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
//       CHECK:   %[[SV_ARG0:.+]] = subview %[[ARG0]][0, %[[LBY]], %[[LBX]], 0]
//       CHECK:   %[[SV_RET0:.+]] = subview %[[RET0]][0, %[[LBY]], %[[LBX]], 0]
//       CHECK:   linalg.pooling_max
//  CHECK-SAME:     %[[SV_ARG0]], %[[ARG1]], %[[SV_RET0]]
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
//       CHECK:   %[[T1:.+]] = addi %[[DIM0]], %[[C3]]
//   CHECK-DAG:   %[[NBY:.+]] = divi_signed %[[T1]], %[[C4]]
//       CHECK:   %[[T0:.+]] = addi %[[DIM1]], %[[C31]]
//   CHECK-DAG:   %[[NBX:.+]] = divi_signed %[[T0]], %[[C32]]
//       CHECK:   return %[[NBX]], %[[NBY]], %[[C1]]

// -----

hal.executable @matmul_fusion attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @matmul_fusion attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x?xf32>, !flow.dispatch.input<?x?xf32>,
        !flow.dispatch.output<?x?xf32>) -> ()}
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
      func private @matmul_fusion__num_workgroups__
        (!shapex.ranked_shape<[?,?]>, !shapex.ranked_shape<[?,?]>,
         !shapex.ranked_shape<[?,?]>) -> (index, index, index)
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 8)>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0] -> (s0 * 16)>
//       CHECK: func @matmul_fusion()
//  CHECK-SAME:   hal.num_workgroups_fn = @[[NWGFN:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   local_size = dense<[16, 8, 1]>
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder {{.+}} @legacy_io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder {{.+}} @legacy_io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder {{.*}} @legacy_io::@ret0
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-NOT:   scf.parallel
//   CHECK-NOT:   scf.for
//       CHECK:   %[[LBY:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[SV_ARG0:.+]] = subview %[[ARG0]][%[[LBY]], 0]
//       CHECK:   %[[LBX:.+]] = affine.apply #[[MAP3]]()[%[[BIDX]]]
//       CHECK:   %[[SV_ARG1:.+]] = subview %[[ARG1]][0, %[[LBX]]]
//       CHECK:   %[[SV_RET0_1:.+]] = subview %[[RET0]][%[[LBY]], %[[LBX]]]
//       CHECK:   %[[SV_RET0_2:.+]] = subview %[[RET0]][%[[LBY]], %[[LBX]]]
//       CHECK:   linalg.fill(%[[SV_RET0_2]], %{{.+}})
//  CHECK-SAME:     "workgroup"
//       CHECK:   linalg.matmul
//  CHECK-SAME:     "workgroup"
//  CHECK-SAME:     ins(%[[SV_ARG0]], %[[SV_ARG1]]
//  CHECK-SAME:     outs(%[[SV_RET0_1]]

//       CHECK: func private @[[NWGFN]]
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder {{.+}} @legacy_io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder {{.+}} @legacy_io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder {{.*}} @legacy_io::@ret0
//       CHECK:   %[[M:.+]] = dim %[[ARG0]], %[[C0]]
//       CHECK:   %[[N:.+]] = dim %[[ARG1]], %[[C1]]
//       CHECK:   %[[WGY_N:.+]] = addi %[[M]], %{{.+}}
//       CHECK:   %[[WGY:.+]] = divi_signed %[[WGY_N]], %{{.+}}
//       CHECK:   %[[WGX_N:.+]] = addi %[[N]], %{{.+}}
//       CHECK:   %[[WGX:.+]] = divi_signed %[[WGX_N]], %{{.+}}
//       CHECK:   return %[[WGX]], %[[WGY]], %[[C1]]

// -----

hal.executable @conv_no_padding_fusion attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @conv_no_padding_fusion attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x?xf32>, !flow.dispatch.input<?x?xf32>,
        !flow.dispatch.output<?x?xf32>) -> ()}
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.3,
        [Shader], [SPV_KHR_storage_buffer_storage_class]>,
        {max_compute_workgroup_invocations = 128 : i32,
         max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
      func @conv_no_padding_fusion()
        attributes {
          hal.num_workgroups_fn = @conv_no_padding_fusion__num_workgroups__} {
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
      func private @conv_no_padding_fusion__num_workgroups__
        (!shapex.ranked_shape<[?,?,?,?]>, !shapex.ranked_shape<[?,?,?,?]>,
         !shapex.ranked_shape<[?,?,?,?]>) -> (index, index, index)
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 4)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 32)>
//       CHECK: func @conv_no_padding_fusion()
//  CHECK-SAME:   hal.num_workgroups_fn = @[[NWGFN:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   local_size = dense<[32, 4, 1]>
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder {{.*}} @legacy_io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder {{.*}} @legacy_io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder {{.*}} @legacy_io::@ret0
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-DAG:   %[[BIDZ:.+]] = "gpu.block_id"() {dimension = "z"}
//       CHECK:   %[[LBY:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[LBX:.+]] = affine.apply #[[MAP1]]()[%[[BIDX]]]
//       CHECK:   %[[SV_ARG1:.+]] = subview %[[ARG1]]
//  CHECK-SAME:     [%[[BIDZ]], %[[LBY]], %[[LBX]], 0]
//       CHECK:   %[[SV_RET0:.+]] = subview %[[RET0]]
//  CHECK-SAME:     [%[[BIDZ]], %[[LBY]], %[[LBX]], 0]
//       CHECK:   linalg.fill(%[[SV_RET0]], %{{.*}})
//  CHECK-SAME:     "workgroup"
//       CHECK:   linalg.conv
//  CHECK-SAME:     %[[ARG0]], %[[SV_ARG1]], %[[SV_RET0]]
//  CHECK-SAME:     "workgroup"

//       CHECK:   func private @[[NWGFN]]
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = constant 2 : index
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder {{.*}} @legacy_io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder {{.*}} @legacy_io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder {{.*}} @legacy_io::@ret0
//       CHECK:   %[[N:.+]] = dim %[[ARG1]], %[[C0]]
//       CHECK:   %[[R:.+]] = dim %[[RET0]], %[[C1]]
//       CHECK:   %[[S:.+]] = dim %[[RET0]], %[[C2]]
//       CHECK:   %[[WGY_N:.+]] = addi %[[R]], %{{.+}}
//       CHECK:   %[[WGY:.+]] = divi_signed %[[WGY_N]], %{{.+}}
//       CHECK:   %[[WGX_N:.+]] = addi %[[S]], %{{.+}}
//       CHECK:   %[[WGX:.+]] = divi_signed %[[WGX_N]], %{{.+}}
//       CHECK:   return %[[WGX]], %[[WGY]], %[[N]]

// -----

hal.executable @three_op_fusion attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @three_op_fusion attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x?xf32>, !flow.dispatch.input<?x?xf32>,
        !flow.dispatch.output<?x?xf32>) -> ()}
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.3,
        [Shader], [SPV_KHR_storage_buffer_storage_class]>,
        {max_compute_workgroup_invocations = 128 : i32,
         max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
      func @three_op_fusion()
        attributes {
          hal.num_workgroups_fn = @three_op_fusion__num_workgroups__} {
        %cst = constant 0.000000e+00 : f32
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %0 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@arg0, operand_result_index = 0 : i32}
          : memref<?x?xf32>
        %1 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@arg1, operand_result_index = 1 : i32}
          : memref<?x?xf32>
        %d0 = dim %0, %c0 : memref<?x?xf32>
        %d1 = dim %1, %c1 : memref<?x?xf32>
        %2 = alloc(%d0, %d1) : memref<?x?xf32>
        %3 =  iree.placeholder for "interface buffer"
          {binding = @legacy_io::@arg2, operand_result_index = 2 : i32}
          : memref<?xf32>
        %4 =  iree.placeholder for "interface buffer"
          {binding = @legacy_io::@ret0, operand_result_index = 3 : i32}
          : memref<?x?xf32>
        linalg.fill(%2, %cst) : memref<?x?xf32>, f32
        linalg.matmul ins(%0, %1 : memref<?x?xf32>, memref<?x?xf32>)
          outs(%2 : memref<?x?xf32>)
        linalg.generic
          {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                            affine_map<(d0, d1) -> (d1)>,
                            affine_map<(d0, d1) -> (d0, d1)>],
           iterator_types = ["parallel", "parallel"]}
          ins(%2, %3 : memref<?x?xf32>, memref<?xf32>)
          outs(%4 : memref<?x?xf32>) {
          ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32) :
            %5 = addf %arg0, %arg1 : f32
            linalg.yield %5 : f32
          }
        return
      }
      func private @three_op_fusion__num_workgroups__
        (!shapex.ranked_shape<[?,?]>, !shapex.ranked_shape<[?,?]>,
         !shapex.ranked_shape<[?]>, !shapex.ranked_shape<[?,?]>)
        -> (index, index, index)
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @arg2, set=0, binding=2, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 8)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (8, s0 * -8 + s1)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 16)>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0, s1] -> (16, s0 * -16 + s1)>
//       CHECK: func @three_op_fusion
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[ALLOC:.+]] = alloc() : memref<8x16xf32, 3>
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder {{.*}} @legacy_io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder {{.*}} @legacy_io::@arg1
//   CHECK-DAG:   %[[M:.+]] = dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[N:.+]] = dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[ARG2:.+]] = iree.placeholder {{.*}} @legacy_io::@arg2
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder {{.*}} @legacy_io::@ret0
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-NOT:   scf.parallel
//   CHECK-NOT:   scf.for
//       CHECK:   %[[LBY:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[TILE_M:.+]] = affine.min #[[MAP1]]()[%[[BIDY]], %[[M]]]
//       CHECK:   %[[LBX:.+]] = affine.apply #[[MAP2]]()[%[[BIDX]]]
//       CHECK:   %[[TILE_N:.+]] = affine.min #[[MAP3]]()[%[[BIDX]], %[[N]]]
//       CHECK:   %[[N_2:.+]] = dim %[[ARG2]], %[[C0]]
//       CHECK:   %[[TILE_N_2:.+]] = affine.min #[[MAP3]]()[%[[BIDX]], %[[N_2]]]
//       CHECK:   %[[SV_ARG2:.+]] = subview %[[ARG2]][%[[LBX]]] [%[[TILE_N_2]]]
//       CHECK:   %[[M_2:.+]] = dim %[[RET0]], %[[C0]]
//       CHECK:   %[[TILE_M_2:.+]] = affine.min #[[MAP1]]()[%[[BIDY]], %[[M_2]]]
//       CHECK:   %[[N_3:.+]] = dim %[[RET0]], %[[C1]]
//       CHECK:   %[[TILE_N_3:.+]] = affine.min #[[MAP3]]()[%[[BIDX]], %[[N_3]]]
//       CHECK:   %[[SV_RET0:.+]] = subview %[[RET0]][%[[LBY]], %[[LBX]]
//  CHECK-SAME:     [%[[TILE_M_2]], %[[TILE_N_3]]]
//       CHECK:   %[[K:.+]] = dim %[[ARG0]], %[[C1]]
//       CHECK:   %[[SV_ARG0:.+]] = subview %[[ARG0]][%[[LBY]], 0]
//  CHECK-SAME:     [%[[TILE_M]], %[[K]]]
//       CHECK:   %[[SV_ARG1:.+]] = subview %[[ARG1]][0, %[[LBX]]]
//  CHECK-SAME:     [%[[K]], %[[TILE_N]]]
//       CHECK:   %[[SV_ALLOC:.+]] = subview %[[ALLOC]][0, 0]
//  CHECK-SAME:     [%[[TILE_M]], %[[TILE_N]]]
//       CHECK:   linalg.fill(%[[SV_ALLOC]], %{{.+}})
//  CHECK-SAME:     "workgroup"
//       CHECK:   linalg.matmul
//  CHECK-SAME:     "workgroup"
//  CHECK-SAME:     ins(%[[SV_ARG0]], %[[SV_ARG1]]
//  CHECK-SAME:       )
//  CHECK-SAME:     outs(%[[SV_ALLOC]]
//  CHECK-SAME:       )
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[SV_ALLOC]], %[[SV_ARG2]]
//  CHECK-SAME:       )
//  CHECK-SAME:     outs(%[[SV_RET0]]
//  CHECK-SAME:       )

// -----

hal.executable @conv_tiled_and_vectorized attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @conv_tiled_and_vectorized attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x?xf32>, !flow.dispatch.input<?x?xf32>,
        !flow.dispatch.output<?x?xf32>) -> ()}
    module attributes {
      spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>
    }  {
      func @conv_tiled_and_vectorized() attributes {hal.num_workgroups_fn = @get_num_workgroups} {
        %cst = constant 0.000000e+00 : f32
        %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<1x112x112x32xf32>
        %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<1x225x225x16xf32>
        %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x3x16x32xf32>
        linalg.fill(%0, %cst) : memref<1x112x112x32xf32>, f32
        linalg.conv(%2, %1, %0) {dilations = [1, 1], strides = [2, 2]} : memref<3x3x16x32xf32>, memref<1x225x225x16xf32>, memref<1x112x112x32xf32>
        return
      }

      func private @get_num_workgroups(!shapex.ranked_shape<[1,225,225,16]>, !shapex.ranked_shape<[3,3,16,32]>, !shapex.ranked_shape<[1,112,112,32]>) -> (index, index, index)

      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
// CHECK-LABEL: func @conv_tiled_and_vectorized()

// CHECK-COUNT-4: vector.transfer_read

// check tiling loop along filter height/width and input channel
// CHECK: scf.for %{{.*}} = %c0 to %c3 step %c1
// CHECK:   scf.for %{{.*}} = %c0 to %c3 step %c1
// CHECK:     scf.for %{{.*}} = %c0 to %c16 step %c4

// CHECK-COUNT-16: vector.contract

// CHECK-COUNT-3: scf.yield
// CHECK-COUNT-4: vector.transfer_write
