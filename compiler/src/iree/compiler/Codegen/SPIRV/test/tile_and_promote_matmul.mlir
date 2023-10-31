// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-tile-and-promote, cse)))))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[128, 128], [16, 4], [0, 0, 32]]>

hal.executable @matmul_f32_256x1024x128 {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<#spirv.vce<v1.5, [Shader], []>, NVIDIA:DiscreteGPU, #spirv.resource_limits<
      max_compute_shared_memory_size = 49152,
      max_compute_workgroup_invocations = 1024,
      max_compute_workgroup_size = [65535, 65535, 65535],
      subgroup_size = 32>>}>) {
    hal.executable.export public @matmul_f32_256x1024x128 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>,
      workgroup_size = [32 : index, 8 : index, 1 : index]
    }
    builtin.module {
      func.func @matmul_f32_256x1024x128() {
        %c1024 = arith.constant 1024 : index
        %c256 = arith.constant 256 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<256x128xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<128x1024xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<256x1024xf32>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : memref<256x1024xf32>
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
            linalg.fill
              ins(%cst : f32) outs(%11 : memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>)
            linalg.matmul {lowering_config = #config}
              ins(%9, %10 : memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>, memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>)
              outs(%11 : memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>)
            linalg.generic {
              indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel"]}
              ins(%11, %8 : memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>, memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>)
              outs(%11 : memref<128x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>) {
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

// CHECK-LABEL: func.func @matmul_f32_256x1024x128()

//  CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
//  CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index

//  CHECK-DAG: %[[MEM_A:.+]] = memref.alloc() : memref<128x32xf32, #gpu.address_space<workgroup>>
//  CHECK-DAG: %[[MEM_B:.+]] = memref.alloc() : memref<32x128xf32, #gpu.address_space<workgroup>>

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

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 256], [1, 8, 8], [0, 0, 0, 16]]>

hal.executable @batch_matmul_16x1024x1024x80 {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader, Float16], []>, AMD:DiscreteGPU, #spirv.resource_limits<
      max_compute_shared_memory_size = 65536,
      max_compute_workgroup_invocations = 1024,
      max_compute_workgroup_size = [1024, 1024, 1024],
      subgroup_size = 64>>}>) {
    hal.executable.export public @batch_matmul_16x1024x1024x80 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>,
      workgroup_size = [32 : index, 8 : index, 1 : index]
    }
    builtin.module {
      func.func @batch_matmul_16x1024x1024x80() {
        %c0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1024 = arith.constant 1024 : index
        %cst = arith.constant 0.111803398 : f32
        %cst_0 = arith.constant 0.000000e+00 : f16
        %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<16x1024x80xf16>
        %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<16x80x1024xf16>
        %8 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<16x1024x1024xf16>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        scf.for %arg0 = %workgroup_id_z to %c16 step %workgroup_count_z {
          %9 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
          %10 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
          scf.for %arg1 = %9 to %c1024 step %10 {
            %11 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_id_x]
            %12 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_count_x]
            scf.for %arg2 = %11 to %c1024 step %12 {
              %subview = memref.subview %8[%arg0, %arg1, %arg2] [1, 64, 256] [1, 1, 1] : memref<16x1024x1024xf16> to memref<1x64x256xf16, strided<[1048576, 1024, 1], offset: ?>>
              %subview_1 = memref.subview %6[%arg0, %arg1, 0] [1, 64, 80] [1, 1, 1] : memref<16x1024x80xf16> to memref<1x64x80xf16, strided<[81920, 80, 1], offset: ?>>
              %subview_2 = memref.subview %7[%arg0, 0, %arg2] [1, 80, 256] [1, 1, 1] : memref<16x80x1024xf16> to memref<1x80x256xf16, strided<[81920, 1024, 1], offset: ?>>
              linalg.fill
                ins(%cst_0 : f16) outs(%subview : memref<1x64x256xf16, strided<[1048576, 1024, 1], offset: ?>>)
              linalg.batch_matmul {lowering_config = #config}
                ins(%subview_1, %subview_2 : memref<1x64x80xf16, strided<[81920, 80, 1], offset: ?>>, memref<1x80x256xf16, strided<[81920, 1024, 1], offset: ?>>)
                outs(%subview : memref<1x64x256xf16, strided<[1048576, 1024, 1], offset: ?>>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
                outs(%subview : memref<1x64x256xf16, strided<[1048576, 1024, 1], offset: ?>>) {
              ^bb0(%out: f16):
                %13 = arith.truncf %cst : f32 to f16
                %14 = arith.mulf %out, %13 : f16
                linalg.yield %14 : f16
              }
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @batch_matmul_16x1024x1024x80()

//  CHECK-NOT: memref.alloc
//  CHECK-DAG: %[[LHS_MEM:.+]] = memref.alloc() : memref<1x64x16xf16, #gpu.address_space<workgroup>>
//  CHECK-DAG: %[[RHS_MEM:.+]] = memref.alloc() : memref<1x16x256xf16, #gpu.address_space<workgroup>>
//  CHECK-NOT: memref.alloc

//      CHECK:       gpu.barrier
//  CHECK-DAG:       memref.copy %{{.+}}, %[[LHS_MEM]]
// CHECK-SAME:           __internal_linalg_transform__ = "copy_to_workgroup_memory"
//  CHECK-DAG:       memref.copy %{{.+}}, %[[RHS_MEM]]
// CHECK-SAME:           __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      CHECK:       gpu.barrier

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[1, 512, 8], [1, 8, 4], [0, 0, 0, 16]]>

hal.executable @batch_matmul_f32_16x4096x40x4096 {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader], []>, AMD:DiscreteGPU, #spirv.resource_limits<
      max_compute_shared_memory_size = 65536,
      max_compute_workgroup_invocations = 1024,
      max_compute_workgroup_size = [1024, 1024, 1024],
      subgroup_size = 64>>}>) {
    hal.executable.export public @batch_matmul_f32_16x4096x40x4096 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>,
      workgroup_size = [2 : index, 64 : index, 1 : index]
    }
    builtin.module {
      func.func @batch_matmul_f32_16x4096x40x4096() {
        %c16 = arith.constant 16 : index
        %c4096 = arith.constant 4096 : index
        %c40 = arith.constant 40 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<16x4096x4096xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<16x4096x40xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<16x4096x40xf32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        scf.for %arg0 = %workgroup_id_z to %c16 step %workgroup_count_z {
          %3 = affine.apply affine_map<()[s0] -> (s0 * 512)>()[%workgroup_id_y]
          %4 = affine.apply affine_map<()[s0] -> (s0 * 512)>()[%workgroup_count_y]
          scf.for %arg1 = %3 to %c4096 step %4 {
            %5 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_x]
            %6 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_count_x]
            scf.for %arg2 = %5 to %c40 step %6 {
              %subview = memref.subview %2[%arg0, %arg1, %arg2] [1, 512, 8] [1, 1, 1] : memref<16x4096x40xf32> to memref<1x512x8xf32, strided<[163840, 40, 1], offset: ?>>
              %subview_0 = memref.subview %0[%arg0, %arg1, 0] [1, 512, 4096] [1, 1, 1] : memref<16x4096x4096xf32> to memref<1x512x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
              %subview_1 = memref.subview %1[%arg0, 0, %arg2] [1, 4096, 8] [1, 1, 1] : memref<16x4096x40xf32> to memref<1x4096x8xf32, strided<[163840, 40, 1], offset: ?>>
              linalg.fill
                ins(%cst : f32) outs(%subview : memref<1x512x8xf32, strided<[163840, 40, 1], offset: ?>>)
              linalg.batch_matmul {lowering_config = #config}
                ins(%subview_0, %subview_1 : memref<1x512x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<1x4096x8xf32, strided<[163840, 40, 1], offset: ?>>)
                outs(%subview : memref<1x512x8xf32, strided<[163840, 40, 1], offset: ?>>)
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @batch_matmul_f32_16x4096x40x4096()

//   CHECK-NOT: memref.alloc()
//  CHECK-DAG: %[[MEM_A:.+]] = memref.alloc() : memref<1x512x16xf32, #gpu.address_space<workgroup>>
//  CHECK-DAG: %[[MEM_B:.+]] = memref.alloc() : memref<1x16x8xf32, #gpu.address_space<workgroup>>
//   CHECK-NOT: memref.alloc()

//      CHECK:       gpu.barrier
//  CHECK-DAG:       memref.copy %{{.+}}, %[[MEM_A]]
// CHECK-SAME:           __internal_linalg_transform__ = "copy_to_workgroup_memory"
//  CHECK-DAG:       memref.copy %{{.+}}, %[[MEM_B]]
// CHECK-SAME:           __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      CHECK:       gpu.barrier
