// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.target(iree-spirv-concretize-tile-among-workgroups))" -iree-spirv-tile-size=16,4,4 -iree-spirv-workgroup-size=4,4,1 %s | IreeFileCheck %s


hal.executable @conv2d attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan_spirv, filter="vulkan*" {
    hal.executable.entry_point @conv2d attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<1x225x225x16xf32>, !flow.dispatch.input<3x3x16x32xf32>, !flow.dispatch.output<1x112x112x32xf32>) -> ()}
    module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, ARM:IntegratedGPU, {}>}  {
      func @conv2d() {
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

// CHECK: #[[MULMAP:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK: #[[MAP0:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0)[s0] -> (9, d0 * -2 + 225)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (16, -d0 + 32)>
// CHECK: #[[MAP3:.+]] = affine_map<(d0)[s0] -> (4, -d0 + 112)>

// CHECK: hal.executable.entry_point @conv2d {{.+}} {
// CHECK:   %[[C2:.+]] = constant 2 : index
// CHECK:   %[[C28_0:.+]] = constant 28 : index
// CHECK:   %[[C28_1:.+]] = constant 28 : index
// CHECK:   hal.return %[[C2]], %[[C28_0]], %[[C28_1]] : index, index, index

// CHECK: func @conv2d()

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

