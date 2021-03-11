// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.target(iree-spirv-concretize-tile-among-workgroups))" -iree-spirv-tile-size=16,4,4 -iree-spirv-workgroup-size=4,4,1 %s | IreeFileCheck %s

hal.executable @conv2d_static_shape attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan_spirv, filter="vulkan*" {
    hal.executable.entry_point @conv2d_static_shape attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<1x225x225x16xf32>, !flow.dispatch.input<3x3x16x32xf32>, !flow.dispatch.output<1x112x112x32xf32>) -> ()}
    module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, ARM:IntegratedGPU, {}>}  {
      func @conv2d_static_shape() {
        %cst = constant 0.000000e+00 : f32
        %c32 = constant 32 : index
        %c112 = constant 112 : index
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<1x225x225x16xf32>
        %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<3x3x16x32xf32>
        %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<1x112x112x32xf32>
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
        scf.for %arg0 = %3 to %c112 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c112 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c32 step %8 {
              %9 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
              %10 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 225)>(%arg0)[%workgroup_size_z]
              %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
              %12 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 225)>(%arg1)[%workgroup_size_y]
              %13 = subview %0[0, %9, %11, 0] [1, %10, %12, 16] [1, 1, 1, 1] : memref<1x225x225x16xf32> to memref<1x?x?x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
              %14 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg2)[%workgroup_size_x]
              %15 = subview %1[0, 0, 0, %arg2] [3, 3, 16, %14] [1, 1, 1, 1] : memref<3x3x16x32xf32> to memref<3x3x16x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
              %16 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg0)[%workgroup_size_z]
              %17 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg1)[%workgroup_size_y]
              %18 = subview %2[0, %arg0, %arg1, %arg2] [1, %16, %17, %14] [1, 1, 1, 1] : memref<1x112x112x32xf32> to memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
              linalg.fill(%18, %cst) : memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>, f32
              linalg.conv_2d_input_nhwc_filter_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%13, %15 : memref<1x?x?x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, memref<3x3x16x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>) outs(%18 : memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>)
            }
          }
        }
        return
      }
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// Check that for a fully static shaped dispatch region, we can:
// 1) Generate static constant workgroup counts,
// 2) Replace hal.interface.workgroup.{size|count} ops with constants,
// 3) Canonicalize loops and subview ops.

// CHECK: #[[MULMAP:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK: #[[MAP0:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0)[s0] -> (9, d0 * -2 + 225)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (16, -d0 + 32)>
// CHECK: #[[MAP3:.+]] = affine_map<(d0)[s0] -> (4, -d0 + 112)>

// CHECK: hal.executable.entry_point @conv2d_static_shape
// CHECK:   %[[C2:.+]] = constant 2 : index
// CHECK:   %[[C28_0:.+]] = constant 28 : index
// CHECK:   %[[C28_1:.+]] = constant 28 : index
// CHECK:   hal.return %[[C2]], %[[C28_0]], %[[C28_1]] : index, index, index

// CHECK: func @conv2d_static_shape()
// CHECK-SAME: spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}

// CHECK: %[[ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK: %[[ID_Y:.+]] = hal.interface.workgroup.id[1] : index
// CHECK: %[[ID_Z:.+]] = hal.interface.workgroup.id[2] : index

// CHECK: %[[Z_MUL_4:.+]] = affine.apply #[[MULMAP]]()[%[[ID_Z]], %c4]
// CHECK: %[[Y_MUL_4:.+]] = affine.apply #[[MULMAP]]()[%[[ID_Y]], %c4]
// CHECK: %[[X_MUL_16:.+]] = affine.apply #[[MULMAP]]()[%[[ID_X]], %c16]

// CHECK: %[[Z_OFFSET:.+]] = affine.apply #[[MAP0]](%[[Z_MUL_4]])
// CHECK: %[[Z_SIZE:.+]] = affine.min #[[MAP1]](%[[Z_MUL_4]])[%c4]
// CHECK: %[[Y_OFFSET:.+]] = affine.apply #[[MAP0]](%[[Y_MUL_4]])
// CHECK: %[[Y_SIZE:.+]] = affine.min #[[MAP1]](%[[Y_MUL_4]])[%c4]

// CHECK: %[[INPUT:.+]] = subview %{{.+}}[0, %[[Z_OFFSET]], %[[Y_OFFSET]], 0] [1, %[[Z_SIZE]], %[[Y_SIZE]], 16] [1, 1, 1, 1] : memref<1x225x225x16xf32> to memref<1x?x?x16xf32, {{.+}}>

// CHECK: %[[X_SIZE:.+]] = affine.min #[[MAP2]](%[[X_MUL_16]])[%c16]

// CHECK: %[[FILTER:.+]] = subview %{{.+}}[0, 0, 0, %[[X_MUL_16]]] [3, 3, 16, %[[X_SIZE]]] [1, 1, 1, 1] : memref<3x3x16x32xf32> to memref<3x3x16x?xf32, {{.+}}>

// CHECK: %[[Z_SIZE:.+]] = affine.min #[[MAP3]](%[[Z_MUL_4]])[%c4]
// CHECK: %[[Y_SIZE:.+]] = affine.min #[[MAP3]](%[[Y_MUL_4]])[%c4]
// CHECK: %[[OUTPUT:.+]] = subview %{{.+}}[0, %[[Z_MUL_4]], %[[Y_MUL_4]], %[[X_MUL_16]]] [1, %[[Z_SIZE]], %[[Y_SIZE]], %[[X_SIZE]]] [1, 1, 1, 1] : memref<1x112x112x32xf32> to memref<1x?x?x?xf32, {{.+}}>

// CHECK: linalg.fill(%[[OUTPUT]], %{{.+}})
// CHECK: linalg.conv_2d_input_nhwc_filter_hwcf {dilations = dense<1> : tensor<2xi64>, is_root_op, strides = dense<2> : tensor<2xi64>} ins(%[[INPUT]], %[[FILTER]] : memref<1x?x?x16xf32, {{.+}}>, memref<3x3x16x?xf32, {{.+}}>) outs(%[[OUTPUT]] : memref<1x?x?x?xf32, {{.+}}>)

// -----

hal.executable @matmul_dynamic_shape attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan_spirv, filter="vulkan*" {
    hal.executable.entry_point @matmul_dynamic_shape attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<1x225x225x16xf32>, !flow.dispatch.input<3x3x16x32xf32>, !flow.dispatch.output<1x112x112x32xf32>) -> ()}
    module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, ARM:IntegratedGPU, {}>}  {
      func @matmul_dynamic_shape() {
        %cst = constant 0.000000e+00 : f32
        %c0 = constant 0 : index
        %0 = hal.interface.load.constant offset = 0 : index
        %1 = hal.interface.load.constant offset = 1 : index
        %2 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<?x?xf32>
        %3 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<?x?xf32>
        %4 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<?x?xf32>
        %5 = hal.interface.load.constant offset = 2 : index
        %6 = hal.interface.load.constant offset = 3 : index
        %7 = hal.interface.load.constant offset = 4 : index
        %8 = hal.interface.load.constant offset = 5 : index
        %9 = hal.interface.load.constant offset = 6 : index
        %10 = hal.interface.load.constant offset = 7 : index
        %11 = shapex.make_ranked_shape %5, %6 : (index, index) -> !shapex.ranked_shape<[?,?]>
        %12 = shapex.tie_shape %2, %11 : memref<?x?xf32>, !shapex.ranked_shape<[?,?]>
        %13 = shapex.make_ranked_shape %7, %8 : (index, index) -> !shapex.ranked_shape<[?,?]>
        %14 = shapex.tie_shape %3, %13 : memref<?x?xf32>, !shapex.ranked_shape<[?,?]>
        %15 = shapex.make_ranked_shape %9, %10 : (index, index) -> !shapex.ranked_shape<[?,?]>
        %16 = shapex.tie_shape %4, %15 : memref<?x?xf32>, !shapex.ranked_shape<[?,?]>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %17 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
        %18 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %17 to %5 step %18 {
          %19 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
          %20 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %19 to %8 step %20 {
            %21 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%arg0)[%5, %workgroup_size_y]
            %22 = subview %12[%arg0, 0] [%21, %6] [1, 1] : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            %23 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%arg1)[%8, %workgroup_size_x]
            %24 = subview %14[0, %arg1] [%7, %23] [1, 1] : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            %25 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%arg0)[%0, %workgroup_size_y]
            %26 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%arg1)[%1, %workgroup_size_x]
            %27 = subview %16[%arg0, %arg1] [%25, %26] [1, 1] : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            linalg.fill(%27, %cst) {__internal_linalg_transform__ = "workgroup"} : memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>, f32
            linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%22, %24 : memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>, memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>) outs(%27 : memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>)
          }
        }
        return
      }
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// Check that for a fully dynamic shaped dispatch region, we can:
// 1) Generate symbolic workgroup counts,
// 2) Replace hal.interface.workgroup.size (but not .count) ops with constants.

// CHECK: #[[DIV16MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK: #[[DIV4MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
// CHECK: #[[MULMAP:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK: #[[YBOUNDMAP:.+]] = affine_map<(d0)[s0, s1] -> (4, -d0 + s0)>
// CHECK: #[[XBOUNDMAP:.+]] = affine_map<(d0)[s0, s1] -> (16, -d0 + s0)>

// CHECK: hal.executable.entry_point @matmul_dynamic_shape
// CHECK: ^{{.+}}(%[[BBARG0:.+]]: index, %[[BBARG1:.+]]: index, %{{.+}}: index):
// CHECK:   %c1 = constant 1 : index
// CHECK:   %[[SIZE0:.+]] = affine.apply #[[DIV16MAP]]()[%[[BBARG0]]]
// CHECK:   %[[SIZE1:.+]] = affine.apply #[[DIV4MAP]]()[%[[BBARG1]]]
// CHECK:   hal.return %[[SIZE0]], %[[SIZE1]], %c1

// CHECK: func @matmul_dynamic_shape()
// CHECK-SAME: spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}

// CHECK: %[[C_DIM0:.+]] = hal.interface.load.constant offset = 0 : index
// CHECK: %[[C_DIM1:.+]] = hal.interface.load.constant offset = 1 : index
// CHECK: %[[A_DIM0:.+]] = hal.interface.load.constant offset = 2 : index
// CHECK: %[[A_DIM1:.+]] = hal.interface.load.constant offset = 3 : index
// CHECK: %[[B_DIM0:.+]] = hal.interface.load.constant offset = 4 : index
// CHECK: %[[B_DIM1:.+]] = hal.interface.load.constant offset = 5 : index

// CHECK: %[[ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK: %[[COUNT_X:.+]] = hal.interface.workgroup.count[0] : index
// CHECK: %[[ID_Y:.+]] = hal.interface.workgroup.id[1] : index
// CHECK: %[[COUNT_Y:.+]] = hal.interface.workgroup.count[1] : index

// CHECK: %[[Y_LB:.+]] = affine.apply #[[MULMAP]]()[%[[ID_Y]], %c4]
// CHECK: %[[Y_STEP:.+]] = affine.apply #[[MULMAP]]()[%[[COUNT_Y]], %c4]
// CHECK: scf.for %[[IV_Y:.+]] = %[[Y_LB]] to %[[A_DIM0]] step %[[Y_STEP]]
// CHECK:   %[[X_LB:.+]] = affine.apply #[[MULMAP]]()[%[[ID_X]], %c16]
// CHECK:   %[[X_STEP:.+]] = affine.apply #[[MULMAP]]()[%[[COUNT_X]], %c16]
// CHECK:   scf.for %[[IV_X:.+]] = %[[X_LB]] to %[[B_DIM1]] step %[[X_STEP]]
// CHECK:     %[[Y_SIZE:.+]] = affine.min #[[YBOUNDMAP]](%[[IV_Y]])[%[[A_DIM0]], %c4]
// CHECK:     %[[A_TILE:.+]] = subview %{{.+}}[%[[IV_Y]], 0] [%[[Y_SIZE]], %[[A_DIM1]]] [1, 1] : memref<?x?xf32> to memref<?x?xf32, {{.+}}>
// CHECK:     %[[X_SIZE:.+]] = affine.min #[[XBOUNDMAP]](%[[IV_X]])[%[[B_DIM1]], %c16]
// CHECK:     %[[B_TILE:.+]] = subview %{{.+}}[0, %[[IV_X]]] [%[[B_DIM0]], %[[X_SIZE]]] [1, 1] : memref<?x?xf32> to memref<?x?xf32, {{.+}}>
// CHECK:     %[[Y_SIZE:.+]] = affine.min #[[YBOUNDMAP]](%[[IV_Y]])[%[[C_DIM0]], %c4]
// CHECK:     %[[X_SIZE:.+]] = affine.min #[[XBOUNDMAP]](%[[IV_X]])[%[[C_DIM1]], %c16]
// CHECK:     %[[C_TILE:.+]] = subview %{{.+}}[%[[IV_Y]], %[[IV_X]]] [%[[Y_SIZE]], %[[X_SIZE]]] [1, 1] : memref<?x?xf32> to memref<?x?xf32, {{.+}}>
// CHECK:     linalg.fill(%[[C_TILE]], %cst) {__internal_linalg_transform__ = "workgroup"} : memref<?x?xf32, {{.+}}>, f32
// CHECK:     linalg.matmul {__internal_linalg_transform__ = "workgroup", is_root_op} ins(%[[A_TILE]], %[[B_TILE]] : memref<?x?xf32, {{.+}}>, memref<?x?xf32, {{.+}}>) outs(%[[C_TILE]] : memref<?x?xf32, {{.+}}>)
