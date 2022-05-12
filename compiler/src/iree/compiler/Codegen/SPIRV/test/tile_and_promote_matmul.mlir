// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-tile-and-promote))))' --cse %s | FileCheck %s

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[128, 128], [16, 4], [0, 0, 32]]>

hal.executable @matmul_256x1024x128 {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spv.target_env = #spv.target_env<#spv.vce<v1.5, [Shader], []>, NVIDIA:DiscreteGPU,
      {max_compute_shared_memory_size = 49152 : i32,
       max_compute_workgroup_invocations = 1024 : i32,
       max_compute_workgroup_size = dense<[2147483647, 65535, 65535]> : vector<3xi32>,
       subgroup_size = 32 : i32}>}> {
    hal.executable.entry_point public @matmul_256x1024x128 ordinal(0) layout(#executable_layout) {
      translation_info = #iree_codegen.translation_info<SPIRVVectorizeWithWorkgroupMemory, workload_per_wg = [128, 128]>,
      workgroup_size = [32 : index, 8 : index, 1 : index]
    }
    builtin.module {
      func.func @matmul_256x1024x128() {
        %c1024 = arith.constant 1024 : index
        %c256 = arith.constant 256 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<256x128xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<128x1024xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : memref<256x1024xf32>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) offset(%c0) alignment(64) : memref<256x1024xf32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %4 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_y]
        %5 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_count_y]
        scf.for %arg0 = %4 to %c256 step %5 {
          %6 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
          %7 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_count_x]
          scf.for %arg1 = %6 to %c1024 step %7 {
            %8 = memref.subview %2[%arg0, %arg1] [128, 128] [1, 1] : memref<256x1024xf32> to memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
            %9 = memref.subview %0[%arg0, 0] [128, 128] [1, 1] : memref<256x128xf32> to memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
            %10 = memref.subview %1[0, %arg1] [128, 128] [1, 1] : memref<128x1024xf32> to memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
            %11 = memref.subview %3[%arg0, %arg1] [128, 128] [1, 1] : memref<256x1024xf32> to memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
            linalg.fill {lowering_config = #config}
              ins(%cst : f32) outs(%11 : memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>)
            linalg.matmul {lowering_config = #config}
              ins(%9, %10 : memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>, memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>)
              outs(%11 : memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>)
            linalg.generic {
              indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel"]}
              ins(%11, %8 : memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>, memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>)
              outs(%11 : memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>)
              attrs =  {lowering_config = #config} {
            ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
              %12 = arith.divf %arg2, %arg3 : f32
              linalg.yield %12 : f32
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @matmul_256x1024x128()

//  CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
//  CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index

//  CHECK-DAG: %[[MEM_A:.+]] = memref.alloc() : memref<128x32xf32, 3>
//  CHECK-DAG: %[[MEM_B:.+]] = memref.alloc() : memref<32x128xf32, 3>

//  CHECK-DAG: %[[BUFFER_A:.+]] = hal.interface.binding.subspan set(0) binding(0) {{.+}} : memref<256x128xf32>
//  CHECK-DAG: %[[BUFFER_B:.+]] = hal.interface.binding.subspan set(0) binding(1) {{.+}} : memref<128x1024xf32>
//  CHECK-DAG: %[[BUFFER_C:.+]] = hal.interface.binding.subspan set(0) binding(3) {{.+}} : memref<256x1024xf32>
//  CHECK-DAG: %[[BUFFER_D:.+]] = hal.interface.binding.subspan set(0) binding(2) {{.+}} : memref<256x1024xf32>

//      CHECK: scf.for
//      CHECK:   scf.for
//      CHECK:     %[[D:.+]] = memref.subview %[[BUFFER_D]]
//      CHECK:     %[[A:.+]] = memref.subview %[[BUFFER_A]]
//      CHECK:     %[[B:.+]] = memref.subview %[[BUFFER_B]]
//      CHECK:     %[[C:.+]] = memref.subview %[[BUFFER_C]]
//      CHECK:     %[[T_ID_X:.+]] = gpu.thread_id  x
//      CHECK:     %[[T_DIM_X:.+]] = gpu.block_dim  x
//      CHECK:     %[[T_ID_Y:.+]] = gpu.thread_id  y
//      CHECK:     %[[T_DIM_Y:.+]] = gpu.block_dim  y
//      CHECK:     %[[T_OFFSET_Y:.+]] = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%[[T_ID_Y]]]
//      CHECK:     %[[T_SIZE_Y:.+]] = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%[[T_DIM_Y]]]

//      CHECK:     scf.for %[[T_IV_Y:.+]] =
//      CHECK:       scf.for %[[T_IV_X:.+]] =
//      CHECK:         %[[VIEW_C:.+]] = memref.subview %[[C]][%[[T_IV_Y]], %[[T_IV_X]]] [16, 4] [1, 1]
//      CHECK:         linalg.fill
// CHECK-SAME:           outs(%[[VIEW_C]]

//      CHECK:     scf.for %[[T_IV_Y:.+]] = %[[C0]] to %[[C128]] step %[[C32]] {
//      CHECK:       %[[VIEW_A:.+]] = memref.subview %[[A]][0, %[[T_IV_Y]]] [128, 32]
//      CHECK:       %[[VIEW_B:.+]] = memref.subview %[[B]][%[[T_IV_Y]], 0] [32, 128]

//      CHECK:       gpu.barrier
//      CHECK:       memref.copy %[[VIEW_A]], %[[MEM_A]]
// CHECK-SAME:           __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      CHECK:       memref.copy %[[VIEW_B]], %[[MEM_B]]
// CHECK-SAME:           __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      CHECK:       gpu.barrier

//      CHECK:       scf.for %[[T_IV_Y:.+]] =
//      CHECK:         scf.for %[[T_IV_X:.+]] =
//      CHECK:           %[[VIEW_A:.+]] = memref.subview %[[MEM_A]][%[[T_IV_Y]], 0] [16, 32]
//      CHECK:           %[[VIEW_B:.+]] = memref.subview %[[MEM_B]][0, %[[T_IV_X]]] [32, 4]
//      CHECK:           %[[VIEW_C:.+]] = memref.subview %[[C]][%[[T_IV_Y]], %[[T_IV_X]]] [16, 4]
//      CHECK:           linalg.matmul
// CHECK-SAME:             ins(%[[VIEW_A]], %[[VIEW_B]]
// CHECK-SAME:             outs(%[[VIEW_C]]

//      CHECK:     scf.for %[[T_IV_Y:.+]] =
//      CHECK:       scf.for %[[T_IV_X:.+]] =
//      CHECK:         %[[VIEW_C:.+]] = memref.subview %[[C]][%[[T_IV_Y]], %[[T_IV_X]]]
//      CHECK:         %[[VIEW_D:.+]] = memref.subview %[[D]][%[[T_IV_Y]], %[[T_IV_X]]]
//      CHECK:         linalg.generic
// CHECK-SAME:           ins(%[[VIEW_C]], %[[VIEW_D]]
// CHECK-SAME:           outs(%[[VIEW_C]]
