// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.variant(iree-spirv-concretize-workgroup-tiles,iree-spirv-tile-and-vectorize))" -canonicalize -cse %s | IreeFileCheck %s

hal.executable @conv_static_shape_f32 attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target="vulkan" {
    hal.executable.entry_point @conv_static_shape_f32 attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, ARM:IntegratedGPU, {}>}  {
      func @conv_static_shape_f32() {
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
              linalg.fill(%cst, %18) {__internal_linalg_transform__ = "workgroup"} : f32, memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
              linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%13, %15 : memref<1x?x?x16xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 810000 + s0 + d1 * 3600 + d2 * 16 + d3)>>, memref<3x3x16x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 512 + d2 * 32 + d3)>>) outs(%18 : memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>)
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

// CHECK-LABEL: func @conv_static_shape_f32()

// For linalg.fill
// CHECK-COUNT-4: vector.transfer_write

// For linalg.conv_2d_input_nhwc_filter_hwcf
// CHECK-COUNT-4: vector.transfer_read

// check tiling loop along filter height/width and input channel
//      CHECK: scf.for %{{.*}} = %c0 to %c3 step %c1
// CHECK-SAME:     -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>)
//      CHECK:   scf.for %{{.*}} = %c0 to %c3 step %c1
// CHECK-SAME:       -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>)
//      CHECK:     scf.for %{{.*}} = %c0 to %c16 step %c4
// CHECK-SAME:         -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>)

// CHECK-COUNT-16: vector.fma

// CHECK-COUNT-3: scf.yield

// For linalg.conv_2d_input_nhwc_filter_hwcf
// CHECK-COUNT-4: vector.transfer_write

// -----

hal.executable @depthwise_conv_static_shape_f32 attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target="vulkan" {
    hal.executable.entry_point @depthwise_conv_static_shape_f32 attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, ARM:IntegratedGPU, {}>}  {
      func @depthwise_conv_static_shape_f32() {
        %cst = constant 0.000000e+00 : f32
        %c96 = constant 96 : index
        %c56 = constant 56 : index
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<1x113x113x96xf32>
        %1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<3x3x1x96xf32>
        %2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<1x56x56x96xf32>
        %3 = linalg.collapse_shape %1 [[0, 1, 2, 3]] : memref<3x3x1x96xf32> into memref<864xf32>
        %4 = linalg.expand_shape %3 [[0, 1, 2]] : memref<864xf32> into memref<3x3x96xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_z, %workgroup_size_z]
        %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_z, %workgroup_size_z]
        scf.for %arg0 = %5 to %c56 step %6 {
          %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %7 to %c56 step %8 {
            %9 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %10 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %9 to %c96 step %10 {
              %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
              %12 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 113)>(%arg0)[%workgroup_size_z]
              %13 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
              %14 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 113)>(%arg1)[%workgroup_size_y]
              %15 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 96)>(%arg2)[%workgroup_size_x]
              %16 = memref.subview %0[0, %11, %13, %arg2] [1, %12, %14, %15] [1, 1, 1, 1] : memref<1x113x113x96xf32> to memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1225824 + s0 + d1 * 10848 + d2 * 96 + d3)>>
              %17 = memref.subview %4[0, 0, %arg2] [3, 3, %15] [1, 1, 1] : memref<3x3x96xf32> to memref<3x3x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 288 + s0 + d1 * 96 + d2)>>
              %18 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 56)>(%arg0)[%workgroup_size_z]
              %19 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 56)>(%arg1)[%workgroup_size_y]
              %20 = memref.subview %2[0, %arg0, %arg1, %arg2] [1, %18, %19, %15] [1, 1, 1, 1] : memref<1x56x56x96xf32> to memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 301056 + s0 + d1 * 5376 + d2 * 96 + d3)>>
              linalg.fill(%cst, %20) {__internal_linalg_transform__ = "workgroup"} : f32, memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 301056 + s0 + d1 * 5376 + d2 * 96 + d3)>>
              linalg.depthwise_conv_2d_input_nhwc_filter_hwc {__internal_linalg_transform__ = "workgroup", dilations = dense<2> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%16, %17 : memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1225824 + s0 + d1 * 10848 + d2 * 96 + d3)>>, memref<3x3x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 288 + s0 + d1 * 96 + d2)>>) outs(%20 : memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 301056 + s0 + d1 * 5376 + d2 * 96 + d3)>>)
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

// CHECK-LABEL: func @depthwise_conv_static_shape_f32()

// For linalg.fill
// CHECK: vector.transfer_write

// For linalg.depthwise_conv_2d_input_nhwc_filter_hwc
// CHECK: vector.transfer_read

// check tiling loop along filter height/width and input channel
//      CHECK:    scf.for %{{.+}} = %c0 to %c3 step %c1
// CHECK-SAME:        -> (vector<4xf32>)
//      CHECK:      scf.for %{{.+}} = %c0 to %c3 step %c1
// CHECK-SAME:          -> (vector<4xf32>)


// CHECK: vector.fma

// CHECK-COUNT-2: scf.yield

// For linalg.depthwise_conv_2d_input_nhwc_filter_hwc
// CHECK: vector.transfer_write
