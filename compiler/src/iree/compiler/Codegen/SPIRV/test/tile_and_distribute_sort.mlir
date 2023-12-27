// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-tile-and-distribute, cse)))))' %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[1, 0, 16], [1, 0, 1]]>
#translation = #iree_codegen.translation_info<SPIRVBaseDistribute>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>
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
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<64x32x128xi32, #hal.descriptor_type<storage_buffer>>
        memref.assume_alignment %0, 64 : memref<64x32x128xi32, #hal.descriptor_type<storage_buffer>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %1 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
        %subview = memref.subview %0[%workgroup_id_y, 0, %1] [1, 32, 64] [1, 1, 1] : memref<64x32x128xi32, #hal.descriptor_type<storage_buffer>> to memref<1x32x64xi32, strided<[4096, 128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        iree_linalg_ext.sort {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 0, 64], [1, 0, 1]]>} dimension(1) outs(%subview : memref<1x32x64xi32, strided<[4096, 128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) {
        ^bb0(%arg0: i32, %arg1: i32):
          %2 = arith.cmpi slt, %arg0, %arg1 : i32
          iree_linalg_ext.yield %2 : i1
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @static_3d_sort()
//       CHECK: %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//       CHECK: %[[WG_OUTPUT:.+]] = memref.subview %[[ARG0]]
//       CHECK: %[[TID_X:.+]] = gpu.thread_id x
//       CHECK: %[[DIM_X:.+]] = gpu.block_dim x
//       CHECK: %[[TID_Y:.+]] = gpu.thread_id y
//       CHECK: %[[DIM_Y:.+]] = gpu.block_dim y
//       CHECK: scf.for %[[IV_Y:.+]] = %[[TID_Y]] to %{{.+}} step %[[DIM_Y]]
//       CHECK:   scf.for %[[IV_X:.+]] = %[[TID_X]] to %{{.+}} step %[[DIM_X]]
//       CHECK:     %[[DEST:.+]] = memref.subview %[[WG_OUTPUT]][%[[IV_Y]], 0, %[[IV_X]]]
//       CHECK:     iree_linalg_ext.sort
//  CHECK-SAME:       dimension(1)
//  CHECK-SAME:       outs(%[[DEST]]
