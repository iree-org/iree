// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(builtin.module(builtin.func(iree-spirv-tile-and-distribute, cse))))' %s | IreeFileCheck %s

hal.executable private @static_3d_sort  {
  hal.interface @io {
    hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1_xw_external, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan_spirv_fb, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @static_3d_sort attributes {
      interface = @io, ordinal = 0 : index,
      translation.info = {passPipeline = 5 : i32, workloadPerWorkgroup = [16, 1]},
      workgroup_size = [16 : index, 1 : index, 1 : index]
    }
    builtin.module {
      builtin.func @static_3d_sort() {
        %c64 = constant 64 : index
        %c128 = constant 128 : index
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<64x32x128xi32>
        %1 = hal.interface.binding.subspan @io::@s0b1_xw_external[%c0] : memref<64x32x128xi32>
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
            linalg.copy(%5, %6) {__internal_linalg_transform__ = "workgroup", lowering.config = {tileSizes = [[1, 0, 16], [], [1, 0, 1]]}} : memref<?x?x?xi32>, memref<1x32x16xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>>
            linalg_ext.sort dimension(1) {__internal_linalg_transform__ = "workgroup", lowering.config = {tileSizes = [[1, 0, 16], [], [1, 0, 1]]}} outs(%7 : memref<?x32x?xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>>)  {
            ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
              %8 = cmpi slt, %arg2, %arg3 : i32
              linalg_ext.yield %8 : i1
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func @static_3d_sort()
//       CHECK: %[[ARG0:.+]] = hal.interface.binding.subspan @io::@s0b0_ro_external
//       CHECK: %[[ARG1:.+]] = hal.interface.binding.subspan @io::@s0b1_xw_external
//       CHECK: scf.for
//       CHECK:   scf.for
//       CHECK:     %[[WG_INPUT:.+]] = memref.subview %[[ARG0]]
//       CHECK:     %[[WG_INPUT_CAST:.+]] = memref.cast %[[WG_INPUT]]
//       CHECK:     %[[WG_OUTPUT:.+]] = memref.subview %[[ARG1]]
//       CHECK:     %[[TID_X:.+]] = "gpu.thread_id"() {dimension = "x"}
//       CHECK:     %[[TID_Y:.+]] = "gpu.thread_id"() {dimension = "y"}
//       CHECK:     %[[COPY_SOURCE:.+]] = memref.subview %[[WG_INPUT_CAST]][%[[TID_Y]], 0, %[[TID_X]]]
//       CHECK:     %[[COPY_DEST:.+]] = memref.subview %[[WG_OUTPUT]][%[[TID_Y]], 0, %[[TID_X]]]
//       CHECK:     linalg.copy(%[[COPY_SOURCE]], %[[COPY_DEST]])
//       CHECK:     %[[T_OUTPUT_CAST:.+]] = memref.cast %[[COPY_DEST]]
//       CHECK:     linalg_ext.sort dimension(1)
//  CHECK-SAME:       outs(%[[T_OUTPUT_CAST]]
