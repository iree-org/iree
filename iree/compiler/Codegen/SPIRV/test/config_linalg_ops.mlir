// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true}))' %s | IreeFileCheck %s

hal.executable @tensor_insert {
  hal.executable.variant @vulkan_spirv_fb, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, ARM:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    }> {
    hal.executable.entry_point @tensor_insert_slice attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      builtin.func @tensor_insert_slice() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:?x?xi32>
        %1 = hal.interface.load.constant offset = 0 : index
        %2 = hal.interface.load.constant offset = 1 : index
        %3 = hal.interface.binding.subspan @io::@s0b1_xw_external[%c0] : !flow.dispatch.tensor<writeonly:?x?xi32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %4 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_id_y]
        %5 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_count_y]
        %d0 = hal.interface.load.constant offset = 2 : index
        %d1 = hal.interface.load.constant offset = 2 : index
        scf.for %arg0 = %4 to %d0 step %5 {
          %6 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %d0]
          %7 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_x, %workgroup_id_x]
          %8 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_x, %workgroup_count_x]
          scf.for %arg1 = %7 to %d1 step %8 {
            %9 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %d1]
            %10 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1], sizes = [%6, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xi32> -> tensor<?x?xi32>
            %11 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg0)[%1]
            %12 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg1)[%2]
            flow.dispatch.tensor.store %10, %3, offsets = [%11, %12], sizes = [%6, %9], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:?x?xi32>
          }
        }
        return
      }
      hal.interface @io attributes {push_constants = 2 : index, sym_visibility = "private"} {
        hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b1_xw_external, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//      CHECK: hal.executable.entry_point public @tensor_insert_slice
// CHECK-SAME:   translation.info = {passPipeline = "SPIRVDistribute", workloadPerWorkgroup = [64, 1]}
// CHECK-NEXT:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[NWGSX:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
//      CHECK:   hal.return %[[NWGSX]], %[[ARG1]], %[[C1]]

// -----

hal.executable @tensor_insert {
  hal.executable.variant @vulkan_spirv_fb, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, ARM:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    }> {
    hal.executable.entry_point @tensor_insert_slice attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      builtin.func @tensor_insert_slice() {
        %c0 = arith.constant 0 : index
        %d0 = hal.interface.load.constant offset = 0 : index
        %d1 = hal.interface.load.constant offset = 1 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<?x?xi32>{%d0, %d1}
        %1 = hal.interface.binding.subspan @io::@s0b1_xw_external[%c0] : memref<?x?xi32>{%d0, %d1}
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %2 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_id_y]
        %3 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_count_y]
        scf.for %arg0 = %2 to %d0 step %3 {
          %4 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %d0]
          %5 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_x, %workgroup_id_x]
          %6 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_x, %workgroup_count_x]
          scf.for %arg1 = %5 to %d1 step %6 {
            %7 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %d1]
            %8 = memref.subview %0[%arg0, %arg1] [%4, %7] [1, 1] : memref<?x?xi32> to memref<?x?xi32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            %9 = affine.apply affine_map<(d0) -> (d0 + 4)>(%arg0)
            %10 = affine.apply affine_map<(d0) -> (d0 + 3)>(%arg1)
            %11 = memref.subview %1[%9, %10] [%4, %7] [1, 1] : memref<?x?xi32> to memref<?x?xi32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            linalg.copy(%8, %11) : memref<?x?xi32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>, memref<?x?xi32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>> 
          }
        }
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = {tileSizes = {{\[}}[1, 16], [1, 1]{{\]}}}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//      CHECK: hal.executable.entry_point public @tensor_insert_slice
// CHECK-SAME:   translation.info = {passPipeline = "SPIRVDistribute", workloadPerWorkgroup = [16, 1]}
// CHECK-NEXT:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[NWGSX:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
//      CHECK:   hal.return %[[NWGSX]], %[[ARG1]], %[[C1]]
//      CHECK:   linalg.copy
// CHECK-SAME:     lowering.config = #[[CONFIG]]
