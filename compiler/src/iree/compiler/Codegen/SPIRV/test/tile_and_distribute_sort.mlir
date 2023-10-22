// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-tile-and-distribute, cse)))))' %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[1, 0, 16], [1, 0, 1]]>
#translation = #iree_codegen.translation_info<SPIRVBaseDistribute>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_3d_sort  {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export @static_3d_sort layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [16 : index, 1 : index, 1 : index]
    }
    builtin.module {
      func.func @static_3d_sort() {
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<64x32x128xi32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<64x32x128xi32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        scf.for %arg0 = %workgroup_id_y to %c64 step %workgroup_count_y {
          %2 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_x]
          %3 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_x]
          scf.for %arg1 = %2 to %c128 step %3 {
            %4 = memref.subview %0[%arg0, 0, %arg1] [1, 32, 16] [1, 1, 1] : memref<64x32x128xi32> to memref<1x32x16xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>>
            %5 = memref.cast %4 : memref<1x32x16xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>> to memref<?x?x?xi32>
            %6 = memref.subview %1[%arg0, 0, %arg1] [1, 32, 16] [1, 1, 1] : memref<64x32x128xi32> to memref<1x32x16xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>>
            %7 = memref.cast %6 : memref<1x32x16xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>> to memref<?x32x?xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
              ins(%5 : memref<?x?x?xi32>)
              outs(%6 : memref<1x32x16xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>>) {
              ^bb0(%arg4: i32, %s: i32):  // no predecessors
                linalg.yield %arg4 : i32
            }
            iree_linalg_ext.sort {lowering_config = #config} dimension(1) outs(%7 : memref<?x32x?xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>>)  {
            ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
              %8 = arith.cmpi slt, %arg2, %arg3 : i32
              iree_linalg_ext.yield %8 : i1
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @static_3d_sort()
//       CHECK: %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//       CHECK: %[[ARG1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//       CHECK: scf.for
//       CHECK:   scf.for
//       CHECK:     %[[WG_INPUT:.+]] = memref.subview %[[ARG0]]
//       CHECK:     %[[WG_OUTPUT:.+]] = memref.subview %[[ARG1]]
//       CHECK:     %[[TID_X:.+]] = gpu.thread_id x
//       CHECK:     %[[DIM_X:.+]] = gpu.block_dim x
//       CHECK:     %[[TID_Y:.+]] = gpu.thread_id y
//       CHECK:     %[[DIM_Y:.+]] = gpu.block_dim y
//       CHECK:     scf.for %[[IV_Y:.+]] = %[[TID_Y]] to %{{.+}} step %[[DIM_Y]]
//       CHECK:       scf.for %[[IV_X:.+]] = %[[TID_X]] to %{{.+}} step %[[DIM_X]]
//       CHECK:         %[[COPY_SOURCE:.+]] = memref.subview %[[WG_INPUT]][%[[IV_Y]], 0, %[[IV_X]]]
//       CHECK:         %[[COPY_DEST:.+]] = memref.subview %[[WG_OUTPUT]][%[[IV_Y]], 0, %[[IV_X]]]
//       CHECK:         linalg.generic {{.*}} ins(%[[COPY_SOURCE]] {{.*}} outs(%[[COPY_DEST]]
//       CHECK:     scf.for %[[IV_Y:.+]] = %[[TID_Y]] to %{{.+}} step %[[DIM_Y]]
//       CHECK:       scf.for %[[IV_X:.+]] = %[[TID_X]] to %{{.+}} step %[[DIM_X]]
//       CHECK:         %[[COPY_DEST:.+]] = memref.subview %[[WG_OUTPUT]][%[[IV_Y]], 0, %[[IV_X]]]
//       CHECK:         %[[T_OUTPUT_CAST:.+]] = memref.cast %[[COPY_DEST]]
//       CHECK:         iree_linalg_ext.sort
//  CHECK-SAME:           dimension(1)
//  CHECK-SAME:           outs(%[[T_OUTPUT_CAST]]
