// RUN: iree-opt -split-input-file  -iree-spirv-workgroup-tile-size=4,16 -iree-spirv-workgroup-size=4,4,1 -pass-pipeline="hal.executable(hal.executable.variant(iree-spirv-concretize-workgroup-tiles))" -canonicalize -cse  %s | IreeFileCheck %s

hal.executable @matmul_dynamic_shape attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan_spirv, filter="vulkan*" {
    hal.executable.entry_point @matmul_dynamic_shape attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, ARM:IntegratedGPU, {}>}  {
      func @matmul_dynamic_shape() {
        %cst = constant 0.000000e+00 : f32
        %c0 = constant 0 : index
        %0 = hal.interface.load.constant offset = 0 : index
        %1 = hal.interface.load.constant offset = 1 : index
        %2 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x?xf32>
        %3 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<?x?xf32>
        %4 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x?xf32>
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
            %22 = memref.subview %12[%arg0, 0] [%21, %6] [1, 1] : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            %23 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%arg1)[%8, %workgroup_size_x]
            %24 = memref.subview %14[0, %arg1] [%7, %23] [1, 1] : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            %25 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%arg0)[%0, %workgroup_size_y]
            %26 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%arg1)[%1, %workgroup_size_x]
            %27 = memref.subview %16[%arg0, %arg1] [%25, %26] [1, 1] : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            linalg.fill(%cst, %27) {__internal_linalg_transform__ = "workgroup"} : f32, memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%22, %24 : memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>, memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>) outs(%27 : memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>)
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

// Check that for a fully dynamic shaped dispatch region, we can:
// 1) Generate symbolic workgroup counts,
// 2) Replace hal.interface.workgroup.size (but not .count) ops with constants.

//  CHECK-DAG: #[[DIV16MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//  CHECK-DAG: #[[DIV4MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//  CHECK-DAG: #[[MUL16MAP:.+]] = affine_map<()[s0] -> (s0 * 16)>
//  CHECK-DAG: #[[MUL4MAP:.+]] = affine_map<()[s0] -> (s0 * 4)>
//  CHECK-DAG: #[[YBOUNDMAP:.+]] = affine_map<(d0)[s0] -> (4, -d0 + s0)>
//  CHECK-DAG: #[[XBOUNDMAP:.+]] = affine_map<(d0)[s0] -> (16, -d0 + s0)>

//      CHECK: hal.executable.entry_point @matmul_dynamic_shape
//      CHECK: ^{{.+}}(%[[BBARG0:.+]]: index, %[[BBARG1:.+]]: index, %{{.+}}: index):
//      CHECK:   %c1 = constant 1 : index
//      CHECK:   %[[SIZE0:.+]] = affine.apply #[[DIV16MAP]]()[%[[BBARG0]]]
//      CHECK:   %[[SIZE1:.+]] = affine.apply #[[DIV4MAP]]()[%[[BBARG1]]]
//      CHECK:   hal.return %[[SIZE0]], %[[SIZE1]], %c1

//      CHECK: func @matmul_dynamic_shape()
// CHECK-SAME: spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}

//      CHECK: %[[C_DIM0:.+]] = hal.interface.load.constant offset = 0 : index
//      CHECK: %[[C_DIM1:.+]] = hal.interface.load.constant offset = 1 : index
//      CHECK: %[[A_DIM0:.+]] = hal.interface.load.constant offset = 2 : index
//      CHECK: %[[A_DIM1:.+]] = hal.interface.load.constant offset = 3 : index
//      CHECK: %[[B_DIM0:.+]] = hal.interface.load.constant offset = 4 : index
//      CHECK: %[[B_DIM1:.+]] = hal.interface.load.constant offset = 5 : index

//      CHECK: %[[ID_X:.+]] = hal.interface.workgroup.id[0] : index
//      CHECK: %[[COUNT_X:.+]] = hal.interface.workgroup.count[0] : index
//      CHECK: %[[ID_Y:.+]] = hal.interface.workgroup.id[1] : index
//      CHECK: %[[COUNT_Y:.+]] = hal.interface.workgroup.count[1] : index

//      CHECK: %[[Y_LB:.+]] = affine.apply #[[MUL4MAP]]()[%[[ID_Y]]]
//      CHECK: %[[Y_STEP:.+]] = affine.apply #[[MUL4MAP]]()[%[[COUNT_Y]]]
//      CHECK: scf.for %[[IV_Y:.+]] = %[[Y_LB]] to %[[A_DIM0]] step %[[Y_STEP]]
//      CHECK:   %[[X_LB:.+]] = affine.apply #[[MUL16MAP]]()[%[[ID_X]]]
//      CHECK:   %[[X_STEP:.+]] = affine.apply #[[MUL16MAP]]()[%[[COUNT_X]]]
//      CHECK:   scf.for %[[IV_X:.+]] = %[[X_LB]] to %[[B_DIM1]] step %[[X_STEP]]
//      CHECK:     %[[Y_SIZE:.+]] = affine.min #[[YBOUNDMAP]](%[[IV_Y]])[%[[A_DIM0]]]
//      CHECK:     %[[A_TILE:.+]] = memref.subview %{{.+}}[%[[IV_Y]], 0] [%[[Y_SIZE]], %[[A_DIM1]]] [1, 1] : memref<?x?xf32> to memref<?x?xf32, {{.+}}>
//      CHECK:     %[[X_SIZE:.+]] = affine.min #[[XBOUNDMAP]](%[[IV_X]])[%[[B_DIM1]]]
//      CHECK:     %[[B_TILE:.+]] = memref.subview %{{.+}}[0, %[[IV_X]]] [%[[B_DIM0]], %[[X_SIZE]]] [1, 1] : memref<?x?xf32> to memref<?x?xf32, {{.+}}>
//      CHECK:     %[[Y_SIZE:.+]] = affine.min #[[YBOUNDMAP]](%[[IV_Y]])[%[[C_DIM0]]]
//      CHECK:     %[[X_SIZE:.+]] = affine.min #[[XBOUNDMAP]](%[[IV_X]])[%[[C_DIM1]]]
//      CHECK:     %[[C_TILE:.+]] = memref.subview %{{.+}}[%[[IV_Y]], %[[IV_X]]] [%[[Y_SIZE]], %[[X_SIZE]]] [1, 1] : memref<?x?xf32> to memref<?x?xf32, {{.+}}>
//      CHECK:     linalg.fill(%cst, %[[C_TILE]])
//      CHECK:     linalg.matmul
// CHECK-SAME:       ins(%[[A_TILE]], %[[B_TILE]]
// CHECK-SAME:       outs(%[[C_TILE]]
