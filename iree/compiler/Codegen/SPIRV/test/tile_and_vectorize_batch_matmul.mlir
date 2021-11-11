// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-set-num-workgroups,builtin.module(builtin.func(iree-spirv-tile-and-distribute,iree-codegen-remove-single-iteration-loop,iree-spirv-vectorize))))' -canonicalize -cse %s | IreeFileCheck %s

#config = #iree_codegen.lowering.config<tile_sizes = [[1, 8, 64], [1, 8, 4], [0, 0, 0, 4]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"SPIRVVectorize", workload_per_wg = [64, 8, 1]>

hal.executable private @batch_matmul_static_shape  {
  hal.interface private @io  {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @batch_matmul_static_shape attributes {
      interface = @io, ordinal = 0 : index,
      workgroup_size = [16: index, 1: index, 1: index],
      translation.info = #translation
    }
    builtin.module {
      func @batch_matmul_static_shape() {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1024 = arith.constant 1024 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<4x1024x1024xf32>
        %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : memref<4x1024x1024xf32>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : memref<4x1024x1024xf32>
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
        scf.for %arg0 = %3 to %c4 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c1024 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c1024 step %8 {
              %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4)>(%arg0)[%workgroup_size_z]
              %10 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1024)>(%arg1)[%workgroup_size_y]
              %11 = memref.subview %0[%arg0, %arg1, 0] [%9, %10, 1024] [1, 1, 1] : memref<4x1024x1024xf32> to memref<?x?x1024xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
              %12 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1024)>(%arg2)[%workgroup_size_x]
              %13 = memref.subview %1[%arg0, 0, %arg2] [%9, 1024, %12] [1, 1, 1] : memref<4x1024x1024xf32> to memref<?x1024x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
              %14 = memref.subview %2[%arg0, %arg1, %arg2] [%9, %10, %12] [1, 1, 1] : memref<4x1024x1024xf32> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
              linalg.batch_matmul {lowering.config = #config}
                ins(%11, %13 : memref<?x?x1024xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>, memref<?x1024x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>)
                outs(%14 : memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>)
            }
          }
        }
        return
      }
      hal.interface private @io  {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

//    CHECK-LABEL: func @batch_matmul_static_shape
//  CHECK-COUNT-8:   vector.transfer_read
//          CHECK:   %[[FOR_RES:.+]]:8 = scf.for
// CHECK-COUNT-12:     vector.transfer_read
// CHECK-COUNT-32:     vector.fma
//      CHECK:         scf.yield
//  CHECK-COUNT-8:    vector.transfer_write %[[FOR_RES]]
//          CHECK:    return

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [[1, 8, 64], [1, 8, 4], [0, 0, 0, 4]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"SPIRVVectorize", workload_per_wg = [64, 8, 1]>

hal.executable private @fused_fill_batch_matmul  {
  hal.interface private @io  {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @fused_fill_batch_matmul attributes {
      interface = @io, ordinal = 0 : index,
      workgroup_size = [16: index, 1: index, 1: index],
      translation.info = #translation
    }
    builtin.module {
      func @fused_fill_batch_matmul() {
        %zero = arith.constant 0.0 : f32
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1024 = arith.constant 1024 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<4x1024x1024xf32>
        %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : memref<4x1024x1024xf32>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : memref<4x1024x1024xf32>
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
        scf.for %arg0 = %3 to %c4 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c1024 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c1024 step %8 {
              %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4)>(%arg0)[%workgroup_size_z]
              %10 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1024)>(%arg1)[%workgroup_size_y]
              %11 = memref.subview %0[%arg0, %arg1, 0] [%9, %10, 1024] [1, 1, 1] : memref<4x1024x1024xf32> to memref<?x?x1024xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
              %12 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1024)>(%arg2)[%workgroup_size_x]
              %13 = memref.subview %1[%arg0, 0, %arg2] [%9, 1024, %12] [1, 1, 1] : memref<4x1024x1024xf32> to memref<?x1024x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
              %14 = memref.subview %2[%arg0, %arg1, %arg2] [%9, %10, %12] [1, 1, 1] : memref<4x1024x1024xf32> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
              linalg.fill(%zero, %14) {lowering.config = #config} : f32, memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
              linalg.batch_matmul {lowering.config = #config}
                ins(%11, %13 : memref<?x?x1024xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>, memref<?x1024x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>)
                outs(%14 : memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>)
            }
          }
        }
        return
      }
      hal.interface private @io  {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

//    CHECK-LABEL: func @fused_fill_batch_matmul
//  CHECK-COUNT-8:   vector.transfer_write
//  CHECK-COUNT-8:   vector.transfer_read
//          CHECK:   %[[FOR_RES:.+]]:8 = scf.for
// CHECK-COUNT-12:     vector.transfer_read
// CHECK-COUNT-32:     vector.fma
//      CHECK:         scf.yield
//  CHECK-COUNT-8:    vector.transfer_write %[[FOR_RES]]
//          CHECK:    return
