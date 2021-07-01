// RUN: iree-opt -split-input-file  -iree-spirv-workgroup-tile-size=0,4,4,16 -iree-spirv-workgroup-size=4,4,1 -pass-pipeline="hal.executable(hal.executable.variant(iree-spirv-concretize-workgroup-tiles))" -canonicalize -cse  %s | IreeFileCheck %s

hal.executable @conv2d_static_shape attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target="vulkan" {
    hal.executable.entry_point @conv2d_static_shape attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, ARM:IntegratedGPU, {}>}  {
      func @conv2d_static_shape() {
        %cst = constant 0.000000e+00 : f32
        %c32 = constant 32 : index
        %c112 = constant 112 : index
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<1x225x225x16xf32>
        %1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<3x3x16x32xf32>
        %2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<1x112x112x32xf32>
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
              %13 = memref.subview %0[0, %9, %11, 0] [1, %10, %12, 16] [1, 1, 1, 1] : memref<1x225x225x16xf32> to memref<1x?x?x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>
              %14 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg2)[%workgroup_size_x]
              %15 = memref.subview %1[0, 0, 0, %arg2] [3, 3, 16, %14] [1, 1, 1, 1] : memref<3x3x16x32xf32> to memref<3x3x16x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>
              %16 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg0)[%workgroup_size_z]
              %17 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg1)[%workgroup_size_y]
              %18 = memref.subview %2[0, %arg0, %arg1, %arg2] [1, %16, %17, %14] [1, 1, 1, 1] : memref<1x112x112x32xf32> to memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
              linalg.fill(%cst, %18) : f32, memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
              linalg.conv_2d_input_nhwc_filter_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%13, %15 : memref<1x?x?x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, memref<3x3x16x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>) outs(%18 : memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>)
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

// Check that for a fully static shaped dispatch region, we can:
// 1) Generate static constant workgroup counts,
// 2) Replace hal.interface.workgroup.{size|count} ops with constants,
// 3) Canonicalize loops and memref.subview ops.

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 4)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 16)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 8)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0] -> (9, s0 * -8 + 225)>
//  CHECK-DAG: #[[MAP5:.+]] = affine_map<()[s0] -> (16, s0 * -16 + 32)>
//  CHECK-DAG: #[[MAP7:.+]] = affine_map<()[s0] -> (4, s0 * -4 + 112)>

//      CHECK: hal.executable.entry_point @conv2d_static_shape
//      CHECK-DAG:   %[[C2:.+]] = constant 2 : index
//      CHECK-DAG:   %[[C28:.+]] = constant 28 : index
//      CHECK:   hal.return %[[C2]], %[[C28]], %[[C28]] : index, index, index

//      CHECK: func @conv2d_static_shape()
// CHECK-SAME: spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}

//  CHECK-DAG: %[[INPUT:.+]] = hal.interface.binding.subspan @io::@arg0
//  CHECK-DAG: %[[FILTER:.+]] = hal.interface.binding.subspan @io::@arg1
//  CHECK-DAG: %[[OUTPUT:.+]] = hal.interface.binding.subspan @io::@ret0

//  CHECK-DAG: %[[ID_X:.+]] = hal.interface.workgroup.id[0] : index
//  CHECK-DAG: %[[ID_Y:.+]] = hal.interface.workgroup.id[1] : index
//  CHECK-DAG: %[[ID_Z:.+]] = hal.interface.workgroup.id[2] : index

//  CHECK-DAG: %[[OUTPUT_OFFSET_Z:.+]] = affine.apply #[[MAP0]]()[%[[ID_Z]]]
//  CHECK-DAG: %[[OUTPUT_OFFSET_Y:.+]] = affine.apply #[[MAP0]]()[%[[ID_Y]]]
//  CHECK-DAG: %[[OUTPUT_OFFSET_X:.+]] = affine.apply #[[MAP1]]()[%[[ID_X]]]
//  CHECK-DAG: %[[INPUT_OFFSET_Z:.+]] = affine.apply #[[MAP2]]()[%[[ID_Z]]]
//  CHECK-DAG: %[[INPUT_SIZE_Z:.+]] = affine.min #[[MAP3]]()[%[[ID_Z]]]
//  CHECK-DAG: %[[INPUT_OFFSET_Y:.+]] = affine.apply #[[MAP2]]()[%[[ID_Y]]]
//  CHECK-DAG: %[[INPUT_SIZE_Y:.+]] = affine.min #[[MAP3]]()[%[[ID_Y]]]

//      CHECK: %[[INPUT_VIEW:.+]] = memref.subview %[[INPUT]]
// CHECK-SAME:   [0, %[[INPUT_OFFSET_Z]], %[[INPUT_OFFSET_Y]], 0]
// CHECK-SAME:   [1, %[[INPUT_SIZE_Z]], %[[INPUT_SIZE_Y]], 16] [1, 1, 1, 1]
// CHECK-SAME:   memref<1x225x225x16xf32> to memref<1x?x?x16xf32, {{.+}}>

//      CHECK: %[[OUTPUT_SIZE_X:.+]] = affine.min #[[MAP5]]()[%[[ID_X]]]
//      CHECK: %[[FILTER_VIEW:.+]] = memref.subview %[[FILTER]]
// CHECK-SAME:    [0, 0, 0, %[[OUTPUT_OFFSET_X]]] [3, 3, 16, %[[OUTPUT_SIZE_X]]]
// CHECK-SAME:    memref<3x3x16x32xf32> to memref<3x3x16x?xf32, {{.+}}>

//  CHECK-DAG: %[[OUTPUT_SIZE_Z:.+]] = affine.min #[[MAP7]]()[%[[ID_Z]]]
//  CHECK-DAG: %[[OUTPUT_SIZE_Y:.+]] = affine.min #[[MAP7]]()[%[[ID_Y]]]
//      CHECK: %[[OUTPUT_VIEW:.+]] = memref.subview %[[OUTPUT]]
// CHECK-SAME:   [0, %[[OUTPUT_OFFSET_Z]], %[[OUTPUT_OFFSET_Y]], %[[OUTPUT_OFFSET_X]]]
// CHECK-SAME:   [1, %[[OUTPUT_SIZE_Z]], %[[OUTPUT_SIZE_Y]], %[[OUTPUT_SIZE_X]]]
// CHECK-SAME:   memref<1x112x112x32xf32> to memref<1x?x?x?xf32, {{.+}}>

//      CHECK: linalg.fill(%{{.+}}, %[[OUTPUT_VIEW]])
//      CHECK: linalg.conv_2d_input_nhwc_filter_hwcf
// CHECK-SAME:   ins(%[[INPUT_VIEW]], %[[FILTER_VIEW]] : memref<1x?x?x16xf32, #map{{[0-9]+}}>, memref<3x3x16x?xf32, #map{{[0-9]+}}>)
// CHECK-SAME:   outs(%[[OUTPUT_VIEW]] : memref<1x?x?x?xf32, #map{{[0-9]+}}>)
