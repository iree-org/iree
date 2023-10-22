// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-tile-and-distribute)))))' %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[1, 16], [1, 1]]>
#translation = #iree_codegen.translation_info<SPIRVBaseDistribute>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @static_scatter_update_slice  {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb">) {
    hal.executable.export @static_scatter_update_slice layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [16 : index, 1 : index, 1 : index]
    }
    builtin.module {
      func.func @static_scatter_update_slice() {
        %c40 = arith.constant 40 : index
        %c500 = arith.constant 500 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<40x500xi32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<40x1xi32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<100x500xi32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        scf.for %arg0 = %workgroup_id_y to %c40 step %workgroup_count_y {
          %3 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_x]
          %4 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_x]
          scf.for %arg1 = %3 to %c500 step %4 {
            %5 = affine.min affine_map<(d0) -> (16, -d0 + 500)>(%arg1)
            %6 = memref.subview %0[%arg0, %arg1] [1, %5] [1, 1] : memref<40x500xi32> to memref<1x?xi32, affine_map<(d0, d1)[s0] -> (d0 * 500 + s0 + d1)>>
            %7 = memref.cast %6 : memref<1x?xi32, affine_map<(d0, d1)[s0] -> (d0 * 500 + s0 + d1)>> to memref<?x?xi32, affine_map<(d0, d1)[s0] -> (d0 * 500 + s0 + d1)>>
            %8 = memref.subview %1[%arg0, 0] [1, 1] [1, 1] : memref<40x1xi32> to memref<1x1xi32, affine_map<(d0, d1)[s0] -> (d0 + s0 + d1)>>
            %9 = memref.cast %8 : memref<1x1xi32, affine_map<(d0, d1)[s0] -> (d0 + s0 + d1)>> to memref<?x1xi32, affine_map<(d0, d1)[s0] -> (d0 + s0 + d1)>>
            %10 = memref.subview %2[0, %arg1] [100, %5] [1, 1] : memref<100x500xi32> to memref<100x?xi32, affine_map<(d0, d1)[s0] -> (d0 * 500 + s0 + d1)>>
            iree_linalg_ext.scatter {lowering_config = #config} dimension_map = [0] unique_indices(true) ins(%7, %9 : memref<?x?xi32, affine_map<(d0, d1)[s0] -> (d0 * 500 + s0 + d1)>>, memref<?x1xi32, affine_map<(d0, d1)[s0] -> (d0 + s0 + d1)>>) outs(%10 : memref<100x?xi32, affine_map<(d0, d1)[s0] -> (d0 * 500 + s0 + d1)>>)  {
            ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
              iree_linalg_ext.yield %arg2 : i32
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @static_scatter_update_slice()
//       CHECK: %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//       CHECK: %[[ARG1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//       CHECK: %[[ARG2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//       CHECK: scf.for
//       CHECK:   scf.for
//       CHECK:     %[[WG_UPDATE:.+]] = memref.subview %[[ARG0]]
//       CHECK:     %[[WG_INDEX:.+]] = memref.subview %[[ARG1]]
//       CHECK:     %[[WG_TARGET:.+]] = memref.subview %[[ARG2]]
//       CHECK:     %[[TID_X:.+]] = gpu.thread_id x
//       CHECK:     %[[DIM_X:.+]] = gpu.block_dim x
//       CHECK:     %[[TID_Y:.+]] = gpu.thread_id y
//       CHECK:     %[[DIM_Y:.+]] = gpu.block_dim y
//       CHECK:     scf.for %[[IV_Y:.+]] = %[[TID_Y]] to %{{.+}} step %[[DIM_Y]]
//       CHECK:       scf.for %[[IV_X:.+]] = %[[TID_X]] to %{{.+}} step %[[DIM_X]]
//       CHECK:         %[[T_UPDATE:.+]] = memref.subview %[[WG_UPDATE]][%[[IV_Y]], %[[IV_X]]] [1, 1] [1, 1]
//       CHECK:         %[[T_UPDATE_CAST:.+]] = memref.cast %[[T_UPDATE]]
//       CHECK:         %[[T_INDEX:.+]] = memref.subview %[[WG_INDEX]][%[[IV_Y]], 0] [1, 1] [1, 1]
//       CHECK:         %[[T_INDEX_CAST:.+]] = memref.cast %[[T_INDEX]]
//       CHECK:         %[[T_TARGET:.+]] = memref.subview %[[WG_TARGET]][0, %[[IV_X]]] [100, 1] [1, 1]
//       CHECK:         %[[T_TARGET_CAST:.+]] = memref.cast %[[T_TARGET]]
//       CHECK:         iree_linalg_ext.scatter
//  CHECK-SAME:           unique_indices(true)
//  CHECK-SAME:           ins(%[[T_UPDATE_CAST]], %[[T_INDEX_CAST]]
//  CHECK-SAME:           outs(%[[T_TARGET_CAST]]
