// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true}))' %s | IreeFileCheck %s

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @tensor_insert {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    }> {
    hal.executable.entry_point @tensor_insert_slice layout(#executable_layout)
    builtin.module {
      builtin.func @tensor_insert_slice() {
        %c0 = arith.constant 0 : index
        %1 = hal.interface.constant.load[0] : index
        %2 = hal.interface.constant.load[1] : index
        %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:?x?xi32>{%1, %2}
        %3 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : !flow.dispatch.tensor<writeonly:?x?xi32>{%1, %2}
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %4 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_id_y]
        %5 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_count_y]
        %d0 = hal.interface.constant.load[2] : index
        %d1 = hal.interface.constant.load[2] : index
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
    }
  }
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistributeCopy", workload_per_wg = [16, 1]>
//      CHECK: hal.executable.entry_point public @tensor_insert_slice
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
//  CHECK-NOT:   hal.return

// -----

#executable_layout = #hal.executable.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @tensor_insert {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    }> {
    hal.executable.entry_point @tensor_insert_slice layout(#executable_layout)
    builtin.module {
      builtin.func @tensor_insert_slice() {
        %c0 = arith.constant 0 : index
        %d0 = hal.interface.constant.load[0] : index
        %d1 = hal.interface.constant.load[1] : index
        %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : memref<?x?xi32>{%d0, %d1}
        %1 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : memref<?x?xi32>{%d0, %d1}
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

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[1, 16], [1, 1]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistribute", workload_per_wg = [16, 1]>
//      CHECK: hal.executable.entry_point public @tensor_insert_slice
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[NWGSX:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
//      CHECK:   hal.return %[[NWGSX]], %[[ARG1]], %[[C1]]
//      CHECK:   linalg.copy
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @tensor_insert {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 64 : i32}>
    }> {
    hal.executable.entry_point @copy layout(#executable_layout)
    builtin.module {
      builtin.func @copy() {
        %c0 = arith.constant 0 : index
        %c224 = arith.constant 224 : index
        %c3 = arith.constant 3 : index
        %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : memref<1x225x225x3xf32>
        %1 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : memref<1x224x224x3xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %2 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_z, %workgroup_id_z]
        %3 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_z, %workgroup_count_z]
        scf.for %arg0 = %2 to %c224 step %3 {
          %4 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 224)>(%arg0)[%workgroup_size_z]
          %5 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_id_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_count_y]
          scf.for %arg1 = %5 to %c224 step %6 {
            %7 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 224)>(%arg1)[%workgroup_size_y]
            %8 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_x, %workgroup_id_x]
            %9 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_x, %workgroup_count_x]
            scf.for %arg2 = %8 to %c3 step %9 {
              %10 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 3)>(%arg2)[%workgroup_size_x]
              %11 = memref.subview %1[0, %arg0, %arg1, %arg2] [1, %4, %7, %10] [1, 1, 1, 1] : memref<1x224x224x3xf32> to memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 150528 + s0 + d1 * 672 + d2 * 3 + d3)>>
              %12 = memref.subview %0[0, %arg0, %arg1, %arg2] [1, %4, %7, %10] [1, 1, 1, 1] : memref<1x225x225x3xf32> to memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 151875 + s0 + d1 * 675 + d2 * 3 + d3)>>
              linalg.copy(%11, %12) : memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 150528 + s0 + d1 * 672 + d2 * 3 + d3)>>, memref<1x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 151875 + s0 + d1 * 675 + d2 * 3 + d3)>>
            }
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[0, 2, 32, 1], [0, 1, 1, 1]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[MAP_Y:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
//  CHECK-DAG: #[[MAP_Z:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistribute", workload_per_wg = [1, 32, 2]>

//      CHECK: hal.executable.entry_point public @copy
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[X:.+]]: index, %[[Y:.+]]: index, %[[Z:.+]]: index)
//  CHECK-DAG:   %[[Y_COUNT:.+]] = affine.apply #[[MAP_Y]]()[%[[Y]]]
//  CHECK-DAG:   %[[Z_COUNT:.+]] = affine.apply #[[MAP_Z]]()[%[[Z]]]
//      CHECK:   hal.return %[[X]], %[[Y_COUNT]], %[[Z_COUNT]]

//      CHECK:   linalg.copy
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0) -> (d0 * 12)>
#map2 = affine_map<(d0)[s0] -> (s0 * 12, d0 * -12 + 24)>
#map3 = affine_map<(d0)[s0] -> (s0, -d0 + 8)>
#map4 = affine_map<(d0)[s0] -> (s0, -d0 + 2)>
#map5 = affine_map<(d0)[s0] -> (-d0 + 2, s0)>
#map6 = affine_map<(d0)[s0] -> (-d0 + 8, s0)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @avg_pool {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 32 : i32}>
    }> {
    hal.executable.entry_point public @avg_pool layout(#executable_layout)
    builtin.module {
      func @avg_pool() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c2 = arith.constant 2 : index
        %c8 = arith.constant 8 : index
        %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:1x24x24x8xf32>
        %1 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : !flow.dispatch.tensor<writeonly:1x2x2x8xf32>
        %2 = linalg.init_tensor [12, 12] : tensor<12x12xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %3 = affine.apply #map0()[%workgroup_id_z, %workgroup_size_z]
        %4 = affine.apply #map0()[%workgroup_count_z, %workgroup_size_z]
        scf.for %arg0 = %3 to %c2 step %4 {
          %5 = affine.apply #map0()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply #map0()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c2 step %6 {
            %7 = affine.apply #map0()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply #map0()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c8 step %8 {
              %9 = affine.apply #map1(%arg0)
              %10 = affine.min #map2(%arg0)[%workgroup_size_z]
              %11 = affine.apply #map1(%arg1)
              %12 = affine.min #map2(%arg1)[%workgroup_size_y]
              %13 = affine.min #map3(%arg2)[%workgroup_size_x]
              %14 = flow.dispatch.tensor.load %0, offsets = [0, %9, %11, %arg2], sizes = [1, %10, %12, %13], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x24x24x8xf32> -> tensor<1x?x?x?xf32>
              %15 = affine.min #map4(%arg0)[%workgroup_size_z]
              %16 = affine.min #map4(%arg1)[%workgroup_size_y]
              %17 = affine.min #map5(%arg0)[%workgroup_size_z]
              %18 = affine.min #map5(%arg1)[%workgroup_size_y]
              %19 = affine.min #map6(%arg2)[%workgroup_size_x]
              %20 = linalg.init_tensor [1, %17, %18, %19] : tensor<1x?x?x?xf32>
              %21 = linalg.fill(%cst, %20) : f32, tensor<1x?x?x?xf32> -> tensor<1x?x?x?xf32>
              %22 = linalg.pooling_nhwc_sum {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : vector<2xi64>, strides = dense<12> : vector<2xi64>} ins(%14, %2 : tensor<1x?x?x?xf32>, tensor<12x12xf32>) outs(%21 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              flow.dispatch.tensor.store %22, %1, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %15, %16, %13], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x2x2x8xf32>
            }
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[0, 2, 2, 0, 0, 8], [0, 1, 1, 0, 0, 1]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistribute", workload_per_wg = [8, 2, 2]>

//      CHECK: hal.executable.entry_point public @avg_pool
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[X:.+]]: index, %[[Y:.+]]: index, %[[Z:.+]]: index)
//  CHECK-DAG:   %[[X_COUNT:.+]] = affine.apply #[[MAP0]]()[%[[X]]]
//  CHECK-DAG:   %[[Y_COUNT:.+]] = affine.apply #[[MAP1]]()[%[[Y]]]
//  CHECK-DAG:   %[[Z_COUNT:.+]] = affine.apply #[[MAP1]]()[%[[Z]]]
//      CHECK:   hal.return %[[X_COUNT]], %[[Y_COUNT]], %[[Z_COUNT]]

//      CHECK:   linalg.pooling_nhwc_sum
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @elementwise {
  hal.executable.variant @vulkan_spirv_fb, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 32 : i32}>
    }> {
    hal.executable.entry_point public @elementwise layout(#executable_layout)
    builtin.module {
      func @elementwise() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c10 = arith.constant 10 : index
        %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:1x10xf32>
        %1 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : !flow.dispatch.tensor<readonly:10xf32>
        %2 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(2) : !flow.dispatch.tensor<writeonly:10xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %3 to %c1 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %5 to %c10 step %6 {
            %7 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1)>(%arg0)[%workgroup_size_y]
            %8 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 10)>(%arg1)[%workgroup_size_x]
            %9 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1], sizes = [%7, %8], strides = [1, 1] : !flow.dispatch.tensor<readonly:1x10xf32> -> tensor<?x?xf32>
            %10 = flow.dispatch.tensor.load %1, offsets = [%arg1], sizes = [%8], strides = [1] : !flow.dispatch.tensor<readonly:10xf32> -> tensor<?xf32>
            %11 = linalg.init_tensor [%8] : tensor<?xf32>
            %12 = linalg.generic {
              indexing_maps = [
                affine_map<(d0, d1) -> (d0, d1)>,
                affine_map<(d0, d1) -> (d1)>,
                affine_map<(d0, d1) -> (d1)>],
              iterator_types = ["parallel", "parallel"]
            } ins(%9, %10 : tensor<?x?xf32>, tensor<?xf32>) outs(%11 : tensor<?xf32>) {
            ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
              %13 = arith.addf %arg2, %arg3 : f32
              linalg.yield %13 : f32
            } -> tensor<?xf32>
            flow.dispatch.tensor.store %12, %2, offsets = [%arg1], sizes = [%8], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:10xf32>
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistribute", workload_per_wg = [32, 1]>
//      CHECK: hal.executable.entry_point public @elementwise
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[NWGSX:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
//      CHECK:   hal.return %[[NWGSX]], %[[ARG1]], %[[C1]]
