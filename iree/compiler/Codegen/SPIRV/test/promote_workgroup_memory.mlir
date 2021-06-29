// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.variant(iree-spirv-tile-and-vectorize,canonicalize,cse))" -iree-spirv-use-workgroup-memory %s | IreeFileCheck %s

hal.executable @matmul_promote_workgroup_memory attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan_spirv, filter="vulkan*" {
    hal.executable.entry_point @matmul_promote_workgroup_memory attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
                        {max_compute_workgroup_invocations = 128 : i32,
                         max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
      func @matmul_promote_workgroup_memory() {
        %c32 = constant 32 : index
        %c50 = constant 50 : index
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<25x50xf32>
        %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : memref<50x75xf32>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : memref<25x75xf32>
        %3 = hal.interface.workgroup.id[0] : index
        %4 = hal.interface.workgroup.id[1] : index
        scf.for %arg0 = %c0 to %c50 step %c32 {
          %5 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%4]
          %6 = affine.min affine_map<()[s0] -> (8, s0 * -8 + 25)>()[%4]
          %7 = affine.min affine_map<(d0) -> (32, -d0 + 50)>(%arg0)
          %8 = memref.subview %0[%5, %arg0] [%6, %7] [1, 1] : memref<25x50xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 50 + s0 + d1)>>
          %9 = affine.min affine_map<(d0) -> (32, -d0 + 50)>(%arg0)
          %10 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%3]
          %11 = affine.min affine_map<()[s0] -> (16, s0 * -16 + 75)>()[%3]
          %12 = memref.subview %1[%arg0, %10] [%9, %11] [1, 1] : memref<50x75xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 75 + s0 + d1)>>
          %13 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%4]
          %14 = affine.min affine_map<()[s0] -> (8, s0 * -8 + 25)>()[%4]
          %15 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%3]
          %16 = affine.min affine_map<()[s0] -> (16, s0 * -16 + 75)>()[%3]
          %17 = memref.subview %2[%13, %15] [%14, %16] [1, 1] : memref<25x75xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 75 + s0 + d1)>>
          linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%8, %12 : memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 50 + s0 + d1)>>, memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 75 + s0 + d1)>>) outs(%17 : memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 75 + s0 + d1)>>)
        }
        return
      }
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// CHECK-LABEL: func @matmul_promote_workgroup_memory()
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@s0b0_ro_external
//   CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan @io::@s0b1_ro_external
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@s0b2_xw_external
//   CHECK-DAG:   %[[ALLOC1:.+]] = memref.alloc() : memref<8x32xf32, 3>
//   CHECK-DAG:   %[[ALLOC2:.+]] = memref.alloc() : memref<32x16xf32, 3>
//       CHECK:   scf.for
//       CHECK:     %[[ARG0SV:.+]] = memref.subview %[[ARG0]]
//       CHECK:     %[[ARG1SV:.+]] = memref.subview %[[ARG1]]
//       CHECK:     %[[RET0SV:.+]] = memref.subview %[[RET0]]
//       CHECK:     %[[SUBVIEW1:.+]] = memref.subview %[[ALLOC1]]
//       CHECK:     %[[SUBVIEW2:.+]] = memref.subview %[[ALLOC2]]
//       CHECK:     linalg.copy(%[[ARG0SV]], %[[SUBVIEW1]])
//  CHECK-SAME:       "copy_to_workgroup_memory"
//       CHECK:     linalg.copy(%[[ARG1SV]], %[[SUBVIEW2]])
//  CHECK-SAME:       "copy_to_workgroup_memory"
//       CHECK:     scf.for
//       CHECK:       scf.for
//   CHECK-DAG:         memref.subview %[[SUBVIEW1]]
//   CHECK-DAG:         memref.subview %[[SUBVIEW2]]
//   CHECK-DAG:         memref.subview %[[RET0SV]]

// -----

hal.executable @conv_promote_workgroup_memory attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan_spirv, filter="vulkan*" {
    hal.executable.entry_point @conv_promote_workgroup_memory attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
                        {max_compute_workgroup_invocations = 128 : i32,
                         max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
      func @conv_promote_workgroup_memory() {
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<3x4x6x14xf32>
        %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : memref<2x15x14x6xf32>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : memref<2x13x11x14xf32>
        %3 = hal.interface.workgroup.id[0] : index
        %4 = hal.interface.workgroup.id[1] : index
        %5 = hal.interface.workgroup.id[2] : index
        %6 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%4]
        %7 = affine.min affine_map<()[s0] -> (6, s0 * -4 + 15)>()[%4]
        %8 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%3]
        %9 = affine.min affine_map<()[s0] -> (35, s0 * -32 + 14)>()[%3]
        %10 = memref.subview %1[%5, %6, %8, 0] [1, %7, %9, 6] [1, 1, 1, 1] : memref<2x15x14x6xf32> to memref<1x?x?x6xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1260 + s0 + d1 * 84 + d2 * 6 + d3)>>
        %11 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%4]
        %12 = affine.min affine_map<()[s0] -> (4, s0 * -4 + 13)>()[%4]
        %13 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%3]
        %14 = affine.min affine_map<()[s0] -> (32, s0 * -32 + 11)>()[%3]
        %15 = memref.subview %2[%5, %11, %13, 0] [1, %12, %14, 14] [1, 1, 1, 1] : memref<2x13x11x14xf32> to memref<1x?x?x14xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 2002 + s0 + d1 * 154 + d2 * 14 + d3)>>
        linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%10, %0 : memref<1x?x?x6xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1260 + s0 + d1 * 84 + d2 * 6 + d3)>>, memref<3x4x6x14xf32>) outs(%15 : memref<1x?x?x14xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 2002 + s0 + d1 * 154 + d2 * 14 + d3)>>)
        return
      }
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// CHECK-LABEL: func @conv_promote_workgroup_memory()
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@s0b0_ro_external
//   CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan @io::@s0b1_ro_external
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@s0b2_xw_external
//   CHECK-DAG:   %[[ALLOC1:.+]] = memref.alloc() : memref<1x6x35x6xf32, 3>
//       CHECK:   %[[ARG1SV:.+]] = memref.subview %[[ARG1]]
//       CHECK:   %[[RET0SV:.+]] = memref.subview %[[RET0]]
//       CHECK:   %[[SUBVIEW1:.+]] = memref.subview %[[ALLOC1]]
//       CHECK:   linalg.copy(%[[ARG1SV]], %[[SUBVIEW1]])
//  CHECK-SAME:      "copy_to_workgroup_memory"
//       CHECK:   scf.for
//       CHECK:     scf.for
//       CHECK:       scf.for
//   CHECK-DAG:         memref.subview %[[SUBVIEW1]]
//   CHECK-DAG:         memref.subview %[[ARG0]]
//   CHECK-DAG:         memref.subview %[[RET0SV]]
