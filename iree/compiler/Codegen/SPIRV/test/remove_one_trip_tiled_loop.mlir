// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(builtin.module(builtin.func(iree-spirv-remove-one-trip-tiled-loop))))' %s | IreeFileCheck %s

#config = #iree_codegen.lowering.config<tile_sizes = [[0, 4, 4, 16], [], [0, 4, 1, 4], [0, 0, 0, 0, 1, 1, 4]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"SPIRVVectorize", workload_per_wg = [16, 4, 4]>
hal.executable private @static_shaped_conv  {
  hal.interface @io {
    hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan_spirv_fb, target = #hal.executable.target<"vulkan", "vulkan-spirv-fb"> {
    hal.executable.entry_point @static_shaped_conv attributes {
      interface = @io, ordinal = 0 : index,
      translation.info = #translation,
      workgroup_size = [4 : index, 4 : index, 1 : index]
    }
    builtin.module {
      builtin.func @static_shaped_conv() {
        %cst = arith.constant 0.000000e+00 : f32
        %c112 = arith.constant 112 : index
        %c32 = arith.constant 32 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<1x225x225x3xf32>
        %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : memref<3x3x3x32xf32>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : memref<1x112x112x32xf32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %3 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_z]
        %4 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_z]
        scf.for %arg0 = %3 to %c112 step %4 {
          %5 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_y]
          %6 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_y]
          scf.for %arg1 = %5 to %c112 step %6 {
            %7 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_x]
            %8 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_x]
            scf.for %arg2 = %7 to %c32 step %8 {
              %9 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
              %10 = affine.min affine_map<(d0) -> (9, d0 * -2 + 225)>(%arg0)
              %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
              %12 = affine.min affine_map<(d0) -> (9, d0 * -2 + 225)>(%arg1)
              %13 = memref.subview %0[0, %9, %11, 0] [1, %10, %12, 3] [1, 1, 1, 1] : memref<1x225x225x3xf32> to memref<1x?x?x3xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 151875 + s0 + d1 * 675 + d2 * 3 + d3)>>
              %14 = affine.min affine_map<(d0) -> (16, -d0 + 32)>(%arg2)
              %15 = memref.subview %1[0, 0, 0, %arg2] [3, 3, 3, %14] [1, 1, 1, 1] : memref<3x3x3x32xf32> to memref<3x3x3x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 288 + s0 + d1 * 96 + d2 * 32 + d3)>>
              %16 = affine.min affine_map<(d0) -> (4, -d0 + 112)>(%arg0)
              %17 = affine.min affine_map<(d0) -> (4, -d0 + 112)>(%arg1)
              %18 = memref.subview %2[0, %arg0, %arg1, %arg2] [1, %16, %17, %14] [1, 1, 1, 1] : memref<1x112x112x32xf32> to memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
              linalg.fill(%cst, %18) {__internal_linalg_transform__ = "workgroup", lowering.config = #config} : f32, memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>
              linalg.conv_2d_nhwc_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, lowering.config = #config, strides = dense<2> : tensor<2xi64>}
                ins(%13, %15 : memref<1x?x?x3xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 151875 + s0 + d1 * 675 + d2 * 3 + d3)>>, memref<3x3x3x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 288 + s0 + d1 * 96 + d2 * 32 + d3)>>)
                outs(%18 : memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 401408 + s0 + d1 * 3584 + d2 * 32 + d3)>>)
            }
          }
        }
        return
      }
      hal.interface private @io  {
        hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 4)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 16)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0) -> (16, -d0 + 32)>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0) -> (4, -d0 + 112)>

//       CHECK: func @static_shaped_conv()
//       CHECK: %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0] : index
//       CHECK: %[[WG_ID_Y:.+]] = hal.interface.workgroup.id[1] : index
//       CHECK: %[[WG_ID_Z:.+]] = hal.interface.workgroup.id[2] : index
//       CHECK: %[[OFFSET_Z:.+]] = affine.apply #[[MAP0]]()[%[[WG_ID_Z]]]
//       CHECK: %[[OFFSET_Y:.+]] = affine.apply #[[MAP0]]()[%[[WG_ID_Y]]]
//       CHECK: %[[OFFSET_X:.+]] = affine.apply #[[MAP1]]()[%[[WG_ID_X]]]
//   CHECK-NOT: scf.for
//   CHECK-DAG: %[[SIZE_Z:.+]] = affine.min #[[MAP3]](%[[OFFSET_Z]])
//   CHECK-DAG: %[[SIZE_Y:.+]] = affine.min #[[MAP3]](%[[OFFSET_Y]])
//   CHECK-DAG: %[[SIZE_X:.+]] = affine.min #[[MAP2]](%[[OFFSET_X]])
//       CHECK: %[[OUTPUT:.+]] = memref.subview %{{.+}}[0, %[[OFFSET_Z]], %[[OFFSET_Y]], %[[OFFSET_X]]] [1, %[[SIZE_Z]], %[[SIZE_Y]], %[[SIZE_X]]]
//       CHECK: linalg.fill(%{{.+}}, %[[OUTPUT]])
//       CHECK: linalg.conv_2d_nhwc_hwcf
//  CHECK-SAME:   outs(%[[OUTPUT]]
