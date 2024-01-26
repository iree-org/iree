// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-tile-and-distribute)))))" %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[2, 256, 4]]>
#translation = #iree_codegen.translation_info<LLVMGPUMatmulSimt, {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#map0 = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<()[s0] -> (s0 * 256)>
#map2 = affine_map<(d0) -> (2, -d0 + 1024)>
#map3 = affine_map<(d0) -> (256, -d0 + 1024)>
#map4 = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
hal.executable private @dot_dispatch_0  {
  hal.executable.variant @cuda target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export @dot_dispatch_0 layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [64 : index, 1 : index, 1 : index]
    }
    builtin.module {
      func.func @dot_dispatch_0() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<1024x1024xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<1024x1024xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<1024x1024xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply #map0()[%workgroup_id_y]
        %4 = affine.apply #map0()[%workgroup_count_y]
        scf.for %arg0 = %3 to %c1024 step %4 {
          %5 = affine.apply #map1()[%workgroup_id_x]
          %6 = affine.apply #map1()[%workgroup_count_x]
          scf.for %arg1 = %5 to %c1024 step %6 {
            %8 = memref.subview %0[%arg0, 0] [2, 1024] [1, 1]
                : memref<1024x1024xf32> to memref<2x1024xf32, #map4>
            %10 = memref.subview %1[0, %arg1] [1024, 256] [1, 1]
                : memref<1024x1024xf32> to memref<1024x256xf32, #map4>
            %11 = memref.subview %2[%arg0, %arg1] [2, 256] [1, 1]
                : memref<1024x1024xf32> to memref<2x256xf32, #map4>
            linalg.fill {lowering_config = #config} ins(%cst : f32) outs(%11 : memref<2x256xf32, #map4>)
            linalg.matmul {lowering_config = #config}
                ins(%8, %10 : memref<2x1024xf32, #map4>, memref<1024x256xf32, #map4>)
                outs(%11 : memref<2x256xf32, #map4>)
          }
        }
        return
      }
    }
  }
}

//   CHECK-LABEL: hal.executable private @dot_dispatch_0
//         CHECK:   hal.executable.variant public @cuda
//     CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
//     CHECK-DAG:  %[[C2:.+]] = arith.constant 2 : index
//     CHECK-DAG:  %[[C4:.+]] = arith.constant 4 : index
//     CHECK-DAG:  %[[C256:.+]] = arith.constant 256 : index
//     CHECK-DAG:  %[[C1024:.+]] = arith.constant 1024 : index
//     CHECK-DAG:  %[[BUFFER0:.+]] = memref.alloc() : memref<4x256xf32, #gpu.address_space<workgroup>>
//     CHECK-DAG:  %[[BUFFER1:.+]] = memref.alloc() : memref<2x4xf32, #gpu.address_space<workgroup>>
//     CHECK-DAG:  %[[BUFFER2:.+]] = memref.alloc() : memref<2x256xf32, #gpu.address_space<workgroup>>
//         CHECK:  scf.for %[[K:.+]] = %[[C0]] to %[[C1024]] step %[[C4]] {
//         CHECK:    gpu.barrier
//         CHECK:    memref.copy {{.*}}, {{.*}} {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<2x4xf32, strided<[1024, 1], offset: ?>> to memref<2x4xf32, #gpu.address_space<workgroup>>
//     CHECK-NOT:    gpu.barrier
//         CHECK:    memref.copy {{.*}}, {{.*}} {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<4x256xf32, strided<[1024, 1], offset: ?>> to memref<4x256xf32, #gpu.address_space<workgroup>>
//         CHECK:    gpu.barrier
//         CHECK:    scf.for %[[IND0:.+]] = %{{.*}} to %[[C2]] step %[[C2]] {
//         CHECK:      scf.for %[[IND1:.+]] = %{{.*}} to %[[C256]] step %[[C256]] {
//     CHECK-DAG:        %[[A:.+]] = memref.subview %[[BUFFER1]][%[[IND0]], 0] [2, 4] [1, 1] : memref<2x4xf32, #gpu.address_space<workgroup>> to memref<2x4xf32, strided<[4, 1], offset: ?>, #gpu.address_space<workgroup>>
//     CHECK-DAG:        %[[B:.+]] = memref.subview %[[BUFFER0]][0, %[[IND1]]] [4, 4] [1, 1] : memref<4x256xf32, #gpu.address_space<workgroup>> to memref<4x4xf32, strided<[256, 1], offset: ?>, #gpu.address_space<workgroup>>
//     CHECK-DAG:        %[[C:.+]] = memref.subview %[[BUFFER2]][%[[IND0]], %[[IND1]]] [2, 4] [1, 1] : memref<2x256xf32, #gpu.address_space<workgroup>> to memref<2x4xf32, strided<[256, 1], offset: ?>, #gpu.address_space<workgroup>>
//         CHECK:        linalg.matmul {__internal_linalg_transform__ = "vectorize", {{.*}}} ins(%[[A]], %[[B]] : memref<2x4xf32, strided<[4, 1], offset: ?>, #gpu.address_space<workgroup>>, memref<4x4xf32, strided<[256, 1], offset: ?>, #gpu.address_space<workgroup>>) outs(%[[C]] : memref<2x4xf32, strided<[256, 1], offset: ?>, #gpu.address_space<workgroup>>)
//         CHECK:    }
//         CHECK:  }
//         CHECK:  gpu.barrier
//         CHECK:  memref.copy {{.*}}, {{.*}} {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<2x256xf32, #gpu.address_space<workgroup>> to memref<2x256xf32,
//         CHECK:  gpu.barrier

// -----

#translation = #iree_codegen.translation_info<LLVMGPUMatmulSimt, {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @batch_matmul_func  {
  hal.executable.variant @cuda target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export @batch_matmul_func layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [8 : index, 8 : index, 1 : index]
    }
builtin.module {
  func.func @batch_matmul_func() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0) : memref<4x32x1024xf32>
    memref.assume_alignment %0, 32 : memref<4x32x1024xf32>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0) : memref<4x1024x64xf32>
    memref.assume_alignment %1, 32 : memref<4x1024x64xf32>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) offset(%c0) : memref<4x32x64xf32>
    memref.assume_alignment %2, 32 : memref<4x32x64xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %workgroup_count_z = hal.interface.workgroup.count[2] : index
    scf.for %arg0 = %workgroup_id_z to %c4 step %workgroup_count_z {
      %3 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_y]
      %4 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_count_y]
      scf.for %arg1 = %3 to %c32 step %4 {
        %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
        %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
        scf.for %arg2 = %5 to %c64 step %6 {
          %7 = memref.subview %0[%arg0, %arg1, 0] [1, 8, 1024] [1, 1, 1] : memref<4x32x1024xf32> to memref<1x8x1024xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 32768 + s0 + d1 * 1024 + d2)>>
          %8 = memref.subview %1[%arg0, 0, %arg2] [1, 1024, 32] [1, 1, 1] : memref<4x1024x64xf32> to memref<1x1024x32xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 65536 + s0 + d1 * 64 + d2)>>
          %9 = memref.subview %2[%arg0, %arg1, %arg2] [1, 8, 32] [1, 1, 1] : memref<4x32x64xf32> to memref<1x8x32xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 2048 + s0 + d1 * 64 + d2)>>
          linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 8, 32, 32]]>} ins(%cst : f32) outs(%9 : memref<1x8x32xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 2048 + s0 + d1 * 64 + d2)>>)
          linalg.batch_matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 8, 32, 32]]>} ins(%7, %8 : memref<1x8x1024xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 32768 + s0 + d1 * 1024 + d2)>>, memref<1x1024x32xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 65536 + s0 + d1 * 64 + d2)>>) outs(%9 : memref<1x8x32xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 2048 + s0 + d1 * 64 + d2)>>)
        }
      }
    }
    return
  }
}
}
}

//         CHECK: #[[$MAP:.*]] = affine_map<()[s0] -> (s0 * 4)>
//   CHECK-LABEL: hal.executable private @batch_matmul_func
//         CHECK:   hal.executable.variant public @cuda
//     CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
//     CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//     CHECK-DAG:   %[[TX:.*]] = gpu.thread_id  x
//     CHECK-DAG:   %[[TY:.*]] = gpu.thread_id  y
//         CHECK:   scf.for %{{.*}} = %[[TY]] to %[[C8]] step %[[C8]] {
//     CHECK-DAG:     %[[TX:.*]] = gpu.thread_id  x
//     CHECK-DAG:     %[[TY:.*]] = gpu.thread_id  y
//         CHECK:     %[[TX4:.*]] = affine.apply #[[$MAP]]()[%[[TX]]]
//         CHECK:     scf.for %[[IND1:.+]] = %[[TX4]] to %[[C32]] step %[[C32]] {
//     CHECK-DAG:       memref.subview
//     CHECK-DAG:       memref.subview
//     CHECK-DAG:       memref.subview
//         CHECK:       linalg.batch_matmul {__internal_linalg_transform__ = "vectorize", {{.*}}} ins(%{{.*}}, %{{.*}} : memref<1x1x32xf32, strided<[256, 32, 1], offset: ?>, #gpu.address_space<workgroup>>, memref<1x32x4xf32, strided<[1024, 32, 1], offset: ?>, #gpu.address_space<workgroup>>) outs(%{{.*}} : memref<1x1x4xf32, strided<[256, 32, 1], offset: ?>, #gpu.address_space<workgroup>>)
//         CHECK:    }
//         CHECK:  }
//         CHECK:  gpu.barrier
//         CHECK:  memref.copy {{.*}}, {{.*}} {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<1x8x32xf32, #gpu.address_space<workgroup>> to memref<1x8x32xf32
//         CHECK:  gpu.barrier

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[2, 32, 4]]>
#translation = #iree_codegen.translation_info<LLVMGPUMatmulSimt, {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#map0 = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#map2 = affine_map<(d0) -> (2, -d0 + 1024)>
#map3 = affine_map<(d0) -> (32, -d0 + 1024)>
#map4 = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
hal.executable private @dot_dispatch_0  {
  hal.executable.variant @cuda target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export @dot_dispatch_0 layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [64 : index, 8 : index, 1 : index]
    }
    builtin.module {
      func.func @dot_dispatch_0() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<1024x1024xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<1024x1024xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<1024x1024xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply #map0()[%workgroup_id_y]
        %4 = affine.apply #map0()[%workgroup_count_y]
        scf.for %arg0 = %3 to %c1024 step %4 {
          %5 = affine.apply #map1()[%workgroup_id_x]
          %6 = affine.apply #map1()[%workgroup_count_x]
          scf.for %arg1 = %5 to %c1024 step %6 {
            %8 = memref.subview %0[%arg0, 0] [2, 1024] [1, 1]
                : memref<1024x1024xf32> to memref<2x1024xf32, #map4>
            %10 = memref.subview %1[0, %arg1] [1024, 32] [1, 1]
                : memref<1024x1024xf32> to memref<1024x32xf32, #map4>
            %11 = memref.subview %2[%arg0, %arg1] [2, 32] [1, 1]
                : memref<1024x1024xf32> to memref<2x32xf32, #map4>
            linalg.fill {lowering_config = #config} ins(%cst : f32) outs(%11 : memref<2x32xf32, #map4>)
            linalg.matmul {lowering_config = #config}
                ins(%8, %10 : memref<2x1024xf32, #map4>, memref<1024x32xf32, #map4>)
                outs(%11 : memref<2x32xf32, #map4>)
          }
        }
        return
      }
    }
  }
}

//   CHECK-LABEL: hal.executable private @dot_dispatch_0
//         CHECK:   hal.executable.variant public @cuda
//     CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
//     CHECK-DAG:  %[[C2:.+]] = arith.constant 2 : index
//     CHECK-DAG:  %[[C4:.+]] = arith.constant 4 : index
//     CHECK-DAG:  %[[C8:.+]] = arith.constant 8 : index
//     CHECK-DAG:  %[[C32:.+]] = arith.constant 32 : index
//     CHECK-DAG:  %[[C64:.+]] = arith.constant 64 : index
//     CHECK-DAG:  %[[C1024:.+]] = arith.constant 1024 : index
//     CHECK-DAG:  %[[BUFFER0:.+]] = memref.alloc() : memref<4x32xf32, #gpu.address_space<workgroup>>
//     CHECK-DAG:  %[[BUFFER1:.+]] = memref.alloc() : memref<2x4xf32, #gpu.address_space<workgroup>>
//     CHECK-DAG:  %[[BUFFER2:.+]] = memref.alloc() : memref<2x32xf32, #gpu.address_space<workgroup>>
//         CHECK:  scf.for %[[K:.+]] = %[[C0]] to %[[C1024]] step %[[C4]] {
//         CHECK:    gpu.barrier
//         CHECK:    memref.copy {{.*}}, {{.*}} {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<2x4xf32, strided<[1024, 1], offset: ?>> to memref<2x4xf32, #gpu.address_space<workgroup>>
//     CHECK-NOT:    gpu.barrier
//         CHECK:    memref.copy {{.*}}, {{.*}} {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<4x32xf32, strided<[1024, 1], offset: ?>> to memref<4x32xf32, #gpu.address_space<workgroup>>
//         CHECK:    gpu.barrier
//         CHECK:    scf.for %[[IND0:.+]] = %{{.*}} to %[[C2]] step %[[C8]] {
//         CHECK:      scf.for %[[IND1:.+]] = %{{.*}} to %[[C32]] step %[[C64]] {
//     CHECK-DAG:        %[[A:.+]] = memref.subview %[[BUFFER1]][%[[IND0]], 0] [1, 4] [1, 1] : memref<2x4xf32, #gpu.address_space<workgroup>> to memref<1x4xf32, strided<[4, 1], offset: ?>, #gpu.address_space<workgroup>>
//     CHECK-DAG:        %[[B:.+]] = memref.subview %[[BUFFER0]][0, %[[IND1]]] [4, 1] [1, 1] : memref<4x32xf32, #gpu.address_space<workgroup>> to memref<4x1xf32, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>
//     CHECK-DAG:        %[[C:.+]] = memref.subview %[[BUFFER2]][%[[IND0]], %[[IND1]]] [1, 1] [1, 1] : memref<2x32xf32, #gpu.address_space<workgroup>> to memref<1x1xf32, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>
//         CHECK:        linalg.matmul {__internal_linalg_transform__ = "vectorize", {{.*}}} ins(%[[A]], %[[B]] : memref<1x4xf32, strided<[4, 1], offset: ?>, #gpu.address_space<workgroup>>, memref<4x1xf32, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>) outs(%[[C]] : memref<1x1xf32, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>)
//         CHECK:    }
//         CHECK:  }
//         CHECK:  gpu.barrier
//         CHECK:  memref.copy {{.*}}, {{.*}} {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<2x32xf32, #gpu.address_space<workgroup>> to memref<2x32xf32
//         CHECK:  gpu.barrier


// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[]]>
#translation = #iree_codegen.translation_info<LLVMGPUVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
// Pure reducion case, skip tiling.
hal.executable @reduction_dispatch {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @predict_dispatch_153 layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [1: index, 1: index, 1: index]
    }
    builtin.module {
      func.func @predict_dispatch_153() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0x7FC00000 : f32
        %cst_0 = arith.constant 0xFF800000 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<1000xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<f32>
        linalg.fill {lowering_config = #config} ins(%cst_0 : f32) outs(%1 : memref<f32>)
        linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%0 : memref<1000xf32>) outs(%1 : memref<f32>) attrs = {lowering_config = #config} {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %2 = arith.cmpf ogt, %arg0, %arg1 : f32
          %3 = arith.select %2, %arg0, %arg1 : f32
          %4 = arith.cmpf uno, %arg0, %arg1 : f32
          %5 = arith.select %4, %cst, %3 : f32
          linalg.yield %5 : f32
        }
        return
      }
    }
  }
}

//      CHECK: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[]{{\]}}>
//      CHECK: hal.executable public @reduction_dispatch
//      CHECK: linalg.fill
// CHECK-SAME:     lowering_config = #[[CONFIG]]
//      CHECK: linalg.generic
// CHECK-SAME:     ins(%{{.*}} : memref<1000xf32>) outs(%{{.*}} : memref<f32>)
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorize>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @conv_dispatch  {
  hal.executable.variant @cuda target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export @conv_dispatch layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [64 : index, 1 : index, 1 : index]
    }
    builtin.module {
      func.func @conv_dispatch() {
        %c56 = arith.constant 56 : index
        %c64 = arith.constant 64 : index
        %c802816 = arith.constant 802816 : index
        %c41664 = arith.constant 41664 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<1x64x56x56xf32>
        memref.assume_alignment %0, 64 : memref<1x64x56x56xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c41664) : memref<64x64x1x1xf32>
        memref.assume_alignment %1, 64 : memref<64x64x1x1xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c802816) : memref<1x64x56x56xf32>
        memref.assume_alignment %2, 64 : memref<1x64x56x56xf32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        scf.for %arg0 = %workgroup_id_z to %c64 step %workgroup_count_z {
          scf.for %arg1 = %workgroup_id_y to %c56 step %workgroup_count_y {
            %3 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_id_x]
            %4 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_count_x]
            scf.for %arg2 = %3 to %c56 step %4 {
              %5 = affine.min affine_map<(d0) -> (256, -d0 + 56)>(%arg2)
              %6 = memref.subview %0[0, 0, %arg1, %arg2] [1, 64, 1, %5] [1, 1, 1, 1] : memref<1x64x56x56xf32> to memref<1x64x1x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 200704 + s0 + d1 * 3136 + d2 * 56 + d3)>>
              %7 = memref.subview %1[%arg0, 0, 0, 0] [1, 64, 1, 1] [1, 1, 1, 1] : memref<64x64x1x1xf32> to memref<1x64x1x1xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 64 + s0 + d1 + d2 + d3)>>
              %8 = memref.subview %2[0, %arg0, %arg1, %arg2] [1, 1, 1, %5] [1, 1, 1, 1] : memref<1x64x56x56xf32> to memref<1x1x1x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 200704 + s0 + d1 * 3136 + d2 * 56 + d3)>>
              linalg.fill{lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 1, 256, 4, 4, 4]]>} ins(%cst : f32) outs(%8 : memref<1x1x1x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 200704 + s0 + d1 * 3136 + d2 * 56 + d3)>>)
              linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 1, 256, 4, 4, 4]]>, strides = dense<1> : vector<2xi64>} ins(%6, %7 : memref<1x64x1x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 200704 + s0 + d1 * 3136 + d2 * 56 + d3)>>, memref<1x64x1x1xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 64 + s0 + d1 + d2 + d3)>>) outs(%8 : memref<1x1x1x?xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 200704 + s0 + d1 * 3136 + d2 * 56 + d3)>>)
            }
          }
        }
        return
      }
    }
  }
}

// Check that the convolution is distributed.
// CHECK-LABEL: func.func @conv_dispatch
//       CHECK:   scf.for
//       CHECK:     scf.for
//       CHECK:       scf.for
//       CHECK:         scf.for
//       CHECK:           linalg.fill
//       CHECK:         scf.for
//       CHECK:           linalg.conv_2d_nchw_fchw


// -----

// Check contract-4d, we currently emit suboptimal code as we don't distribute
// more than 3 dimensions but make sure we emit correct code.
#config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 2, 256, 4]]>
#translation = #iree_codegen.translation_info<LLVMGPUMatmulSimt, {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @contract_4d  {
  hal.executable.variant @cuda target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export @contract_4d layout(#pipeline_layout) attributes {
      workgroup_size = [64 : index, 8 : index, 1 : index]
    }
    builtin.module {
      func.func @contract_4d() {
        %c12 = arith.constant 12 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 8.000000e+00 : f32
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %6 = arith.index_cast %0 : i32 to index
        %12 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%6) : memref<?x?x12x64xf32>{%6, %6}
        %13 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%6) : memref<?x?x12x64xf32>{%6, %6}
        %15 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<?x12x?x?xf32>{%6, %6, %6}
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        scf.for %arg0 = %workgroup_id_z to %c12 step %workgroup_count_z {
          %16 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%workgroup_id_y]
          %17 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%workgroup_count_y]
          scf.for %arg1 = %16 to %6 step %17 {
            %18 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_id_x]
            %19 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_count_x]
            scf.for %arg2 = %18 to %6 step %19 {
              %20 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 2)>(%arg1)[%6]
              %21 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 256)>(%arg2)[%6]
              %22 = memref.subview %15[0, %arg0, %arg1, %arg2] [%6, 1, %20, %21] [1, 1, 1, 1] : memref<?x12x?x?xf32> to memref<?x1x?x?xf32, affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>>
              %24 = memref.subview %12[0, %arg1, %arg0, 0] [%6, %20, 1, 64] [1, 1, 1, 1] : memref<?x?x12x64xf32> to memref<?x?x1x64xf32, affine_map<(d0, d1, d2, d3)[s0, s1] -> (d0 * s1 + s0 + d1 * 768 + d2 * 64 + d3)>>
              %25 = memref.subview %13[0, %arg2, %arg0, 0] [%6, %21, 1, 64] [1, 1, 1, 1] : memref<?x?x12x64xf32> to memref<?x?x1x64xf32, affine_map<(d0, d1, d2, d3)[s0, s1] -> (d0 * s1 + s0 + d1 * 768 + d2 * 64 + d3)>>
              linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 2, 256, 4]]>} ins(%cst_0 : f32) outs(%22 : memref<?x1x?x?xf32, affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%24, %25 : memref<?x?x1x64xf32, affine_map<(d0, d1, d2, d3)[s0, s1] -> (d0 * s1 + s0 + d1 * 768 + d2 * 64 + d3)>>, memref<?x?x1x64xf32, affine_map<(d0, d1, d2, d3)[s0, s1] -> (d0 * s1 + s0 + d1 * 768 + d2 * 64 + d3)>>) outs(%22 : memref<?x1x?x?xf32, affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 2, 256, 4]]>} {
              ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
                %26 = arith.mulf %arg3, %arg4 : f32
                %27 = arith.addf %26, %arg5 : f32
                linalg.yield %27 : f32
              }
            }
          }
        }
        return
      }
    }
  }
}

// Check that we are able to distribute correctly
// CHECK-LABEL: func.func @contract_4d
