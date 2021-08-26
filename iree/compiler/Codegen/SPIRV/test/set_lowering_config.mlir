// RUN: iree-opt -split-input-file -mlir-print-local-scope -pass-pipeline='hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true}))' %s | IreeFileCheck %s

hal.executable @static_1d_sort attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @s0b0_rw_external, set=0, binding=0, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.variant @vulkan_spirv_fb, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @static_1d_sort attributes {interface = @io, ordinal = 0 : index}
    builtin.module attributes {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, ARM:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    } {
      builtin.func @static_1d_sort() {
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_rw_external[%c0] : !flow.dispatch.tensor<readwrite:1000xi32>
        %1 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readwrite:1000xi32> -> tensor<1000xi32>
        %2 = linalg_ext.sort dimension(0) {__internal_linalg_transform__ = "workgroup"} outs(%1 : tensor<1000xi32>)  {
        ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
          %3 = cmpi slt, %arg0, %arg1 : i32
          linalg_ext.yield %3 : i1
        } -> tensor<1000xi32>
        flow.dispatch.tensor.store %2, %0, offsets = [], sizes = [], strides = [] : tensor<1000xi32> -> !flow.dispatch.tensor<readwrite:1000xi32>
        return
      }
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @s0b0_rw_external, set=0, binding=0, type="StorageBuffer", access="Read|Write"
      }
    }
  }
}

// Check that the workgroup size is (1, 1, 1) for serializing the computation.

// CHECK-LABEL: hal.executable.entry_point @static_1d_sort
//  CHECK-SAME:   translation.info = {passPipeline = 6 : i32}
//  CHECK-SAME:   workgroup_size = [1 : index, 1 : index, 1 : index]
//  CHECK-NEXT: ^{{.+}}(%{{.+}}: index, %{{.+}}: index, %{{.+}}: index):
//  CHECK-NEXT:   %[[ONE:.+]] = constant 1 : index
//  CHECK-NEXT:   hal.return %[[ONE]], %[[ONE]], %[[ONE]]

//       CHECK: func @static_1d_sort()
//       CHECK:   linalg_ext.sort
//  CHECK-SAME:     lowering.config = {}

// -----

hal.executable @static_3d_sort attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1_xw_external, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan_spirv_fb, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @static_3d_sort attributes {interface = @io, ordinal = 0 : index}
    builtin.module attributes {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, ARM:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    } {
      builtin.func @static_3d_sort() {
        %c64 = constant 64 : index
        %c128 = constant 128 : index
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<64x32x128xi32>
        %1 = hal.interface.binding.subspan @io::@s0b1_xw_external[%c0] : memref<64x32x128xi32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %2 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_id_y]
        %3 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_count_y]
        scf.for %arg0 = %2 to %c64 step %3 {
          %4 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 64)>(%arg0)[%workgroup_size_y]
          %5 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_x, %workgroup_id_x]
          %6 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_x, %workgroup_count_x]
          scf.for %arg1 = %5 to %c128 step %6 {
            %7 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 128)>(%arg1)[%workgroup_size_x]
            %8 = memref.subview %0[%arg0, 0, %arg1] [%4, 32, %7] [1, 1, 1] : memref<64x32x128xi32> to memref<?x32x?xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>>
            %9 = memref.cast %8 : memref<?x32x?xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>> to memref<?x?x?xi32>
            %10 = memref.subview %1[%arg0, 0, %arg1] [%4, 32, %7] [1, 1, 1] : memref<64x32x128xi32> to memref<?x32x?xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>>
            linalg.copy(%9, %10) : memref<?x?x?xi32>, memref<?x32x?xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>>
            linalg_ext.sort dimension(1) {__internal_linalg_transform__ = "workgroup"} outs(%10 : memref<?x32x?xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>>)  {
            ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
              %11 = cmpi slt, %arg2, %arg3 : i32
              linalg_ext.yield %11 : i1
            }
          }
        }
        return
      }
    }
  }
}

//          CHECK-LABEL: hal.executable.entry_point @static_3d_sort
//           CHECK-SAME:   translation.info = {passPipeline = 5 : i32, workloadPerWorkgroup = [16, 1]}
//           CHECK-SAME:   workgroup_size = [16 : index, 1 : index, 1 : index]
//           CHECK-NEXT: ^{{.+}}(%[[X:.+]]: index, %[[Y:.+]]: index, %{{.+}}: index):
//           CHECK-NEXT:   %[[ONE:.+]] = constant 1 : index
//           CHECK-NEXT:   %[[DIV:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%[[X]]]
//           CHECK-NEXT:   hal.return %[[DIV]], %[[Y]], %[[ONE]]

//                CHECK: func @static_3d_sort()
//                CHECK:   linalg_ext.sort
//  CHECK-SAME{LITERAL}:     lowering.config = {tileSizes = [[1, 0, 16], [], [1, 0, 1]]}
