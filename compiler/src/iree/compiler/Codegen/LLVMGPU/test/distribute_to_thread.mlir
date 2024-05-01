// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-llvmgpu-tile-and-distribute))" %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[2, 256, 4]]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {iree.gpu.target = #iree_gpu.alias_target<"sm_60">}>
#map = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<()[s0] -> (s0 * 256)>
#map2 = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
#translation = #iree_codegen.translation_info<LLVMGPUMatmulSimt workgroup_size = [64, 1, 1], {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
module {
  func.func @dot_dispatch_0() attributes {hal.executable.target = #executable_target_cuda_nvptx_fb, translation_info = #translation} {
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
    %3 = affine.apply #map()[%workgroup_id_y]
    %4 = affine.apply #map()[%workgroup_count_y]
    scf.for %arg0 = %3 to %c1024 step %4 {
      %5 = affine.apply #map1()[%workgroup_id_x]
      %6 = affine.apply #map1()[%workgroup_count_x]
      scf.for %arg1 = %5 to %c1024 step %6 {
        %subview = memref.subview %0[%arg0, 0] [2, 1024] [1, 1] : memref<1024x1024xf32> to memref<2x1024xf32, #map2>
        %subview_0 = memref.subview %1[0, %arg1] [1024, 256] [1, 1] : memref<1024x1024xf32> to memref<1024x256xf32, #map2>
        %subview_1 = memref.subview %2[%arg0, %arg1] [2, 256] [1, 1] : memref<1024x1024xf32> to memref<2x256xf32, #map2>
        linalg.fill {lowering_config = #config} ins(%cst : f32) outs(%subview_1 : memref<2x256xf32, #map2>)
        linalg.matmul {lowering_config = #config} ins(%subview, %subview_0 : memref<2x1024xf32, #map2>, memref<1024x256xf32, #map2>) outs(%subview_1 : memref<2x256xf32, #map2>)
      }
    }
    return
  }
}

//         CHECK:  func.func @dot_dispatch_0()
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

#config = #iree_codegen.lowering_config<tile_sizes = [[1, 8, 32, 32]]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {iree.gpu.target = #iree_gpu.alias_target<"sm_60">}>
#map = affine_map<()[s0] -> (s0 * 8)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#map2 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32768 + s0 + d1 * 1024 + d2)>
#map3 = affine_map<(d0, d1, d2)[s0] -> (d0 * 65536 + s0 + d1 * 64 + d2)>
#map4 = affine_map<(d0, d1, d2)[s0] -> (d0 * 2048 + s0 + d1 * 64 + d2)>
#translation = #iree_codegen.translation_info<LLVMGPUMatmulSimt workgroup_size = [8, 8, 1], {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
module {
  func.func @batch_matmul_func() attributes {hal.executable.target = #executable_target_cuda_nvptx_fb, translation_info = #translation} {
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
      %3 = affine.apply #map()[%workgroup_id_y]
      %4 = affine.apply #map()[%workgroup_count_y]
      scf.for %arg1 = %3 to %c32 step %4 {
        %5 = affine.apply #map1()[%workgroup_id_x]
        %6 = affine.apply #map1()[%workgroup_count_x]
        scf.for %arg2 = %5 to %c64 step %6 {
          %subview = memref.subview %0[%arg0, %arg1, 0] [1, 8, 1024] [1, 1, 1] : memref<4x32x1024xf32> to memref<1x8x1024xf32, #map2>
          %subview_0 = memref.subview %1[%arg0, 0, %arg2] [1, 1024, 32] [1, 1, 1] : memref<4x1024x64xf32> to memref<1x1024x32xf32, #map3>
          %subview_1 = memref.subview %2[%arg0, %arg1, %arg2] [1, 8, 32] [1, 1, 1] : memref<4x32x64xf32> to memref<1x8x32xf32, #map4>
          linalg.fill {lowering_config = #config} ins(%cst : f32) outs(%subview_1 : memref<1x8x32xf32, #map4>)
          linalg.batch_matmul {lowering_config = #config} ins(%subview, %subview_0 : memref<1x8x1024xf32, #map2>, memref<1x1024x32xf32, #map3>) outs(%subview_1 : memref<1x8x32xf32, #map4>)
        }
      }
    }
    return
  }
}

//         CHECK: #[[$MAP:.*]] = affine_map<()[s0] -> (s0 * 4)>
//         CHECK: func.func @batch_matmul_func()
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
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {iree.gpu.target = #iree_gpu.alias_target<"sm_60">}>
#map = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#map2 = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
#translation = #iree_codegen.translation_info<LLVMGPUMatmulSimt workgroup_size = [64, 8, 1], {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
module {
  func.func @dot_dispatch_0() attributes {hal.executable.target = #executable_target_cuda_nvptx_fb, translation_info = #translation} {
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
    %3 = affine.apply #map()[%workgroup_id_y]
    %4 = affine.apply #map()[%workgroup_count_y]
    scf.for %arg0 = %3 to %c1024 step %4 {
      %5 = affine.apply #map1()[%workgroup_id_x]
      %6 = affine.apply #map1()[%workgroup_count_x]
      scf.for %arg1 = %5 to %c1024 step %6 {
        %subview = memref.subview %0[%arg0, 0] [2, 1024] [1, 1] : memref<1024x1024xf32> to memref<2x1024xf32, #map2>
        %subview_0 = memref.subview %1[0, %arg1] [1024, 32] [1, 1] : memref<1024x1024xf32> to memref<1024x32xf32, #map2>
        %subview_1 = memref.subview %2[%arg0, %arg1] [2, 32] [1, 1] : memref<1024x1024xf32> to memref<2x32xf32, #map2>
        linalg.fill {lowering_config = #config} ins(%cst : f32) outs(%subview_1 : memref<2x32xf32, #map2>)
        linalg.matmul {lowering_config = #config} ins(%subview, %subview_0 : memref<2x1024xf32, #map2>, memref<1024x32xf32, #map2>) outs(%subview_1 : memref<2x32xf32, #map2>)
      }
    }
    return
  }
}

//         CHECK: func.func @dot_dispatch_0()
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
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {iree.gpu.target = #iree_gpu.alias_target<"sm_60">}>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#translation = #iree_codegen.translation_info<LLVMGPUVectorize workgroup_size = [1, 1, 1]>
module {
  func.func @predict_dispatch_153() attributes {hal.executable.target = #executable_target_cuda_nvptx_fb, translation_info = #translation} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0x7FC00000 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<1000xf32>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<f32>
    linalg.fill {lowering_config = #config} ins(%cst_0 : f32) outs(%1 : memref<f32>)
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%0 : memref<1000xf32>) outs(%1 : memref<f32>) attrs =  {lowering_config = #config} {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.cmpf ogt, %in, %out : f32
      %3 = arith.select %2, %in, %out : f32
      %4 = arith.cmpf uno, %in, %out : f32
      %5 = arith.select %4, %cst, %3 : f32
      linalg.yield %5 : f32
    }
    return
  }
}

//      CHECK: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[]{{\]}}>
//      CHECK: func.func @predict_dispatch_153()
//      CHECK: linalg.fill
// CHECK-SAME:     lowering_config = #[[CONFIG]]
//      CHECK: linalg.generic
// CHECK-SAME:     ins(%{{.*}} : memref<1000xf32>) outs(%{{.*}} : memref<f32>)
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 1, 256, 4, 4, 4]]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {iree.gpu.target = #iree_gpu.alias_target<"sm_60">}>
#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<(d0) -> (256, -d0 + 56)>
#map2 = affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 200704 + s0 + d1 * 3136 + d2 * 56 + d3)>
#map3 = affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 64 + s0 + d1 + d2 + d3)>
#translation = #iree_codegen.translation_info<LLVMGPUVectorize workgroup_size = [64, 1, 1]>
module {
  func.func @conv_dispatch() attributes {hal.executable.target = #executable_target_cuda_nvptx_fb, translation_info = #translation} {
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
        %3 = affine.apply #map()[%workgroup_id_x]
        %4 = affine.apply #map()[%workgroup_count_x]
        scf.for %arg2 = %3 to %c56 step %4 {
          %5 = affine.min #map1(%arg2)
          %subview = memref.subview %0[0, 0, %arg1, %arg2] [1, 64, 1, %5] [1, 1, 1, 1] : memref<1x64x56x56xf32> to memref<1x64x1x?xf32, #map2>
          %subview_0 = memref.subview %1[%arg0, 0, 0, 0] [1, 64, 1, 1] [1, 1, 1, 1] : memref<64x64x1x1xf32> to memref<1x64x1x1xf32, #map3>
          %subview_1 = memref.subview %2[0, %arg0, %arg1, %arg2] [1, 1, 1, %5] [1, 1, 1, 1] : memref<1x64x56x56xf32> to memref<1x1x1x?xf32, #map2>
          linalg.fill {lowering_config = #config} ins(%cst : f32) outs(%subview_1 : memref<1x1x1x?xf32, #map2>)
          linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, lowering_config = #config, strides = dense<1> : vector<2xi64>} ins(%subview, %subview_0 : memref<1x64x1x?xf32, #map2>, memref<1x64x1x1xf32, #map3>) outs(%subview_1 : memref<1x1x1x?xf32, #map2>)
        }
      }
    }
    return
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

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 2, 256, 4]]>
#translation = #iree_codegen.translation_info<LLVMGPUMatmulSimt workgroup_size = [64, 8, 1], {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {iree.gpu.target = #iree_gpu.alias_target<"sm_60">}>
#map = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<()[s0] -> (s0 * 256)>
#map2 = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
#map3 = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
#map4 = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>
#map5 = affine_map<(d0, d1, d2, d3)[s0, s1] -> (d0 * s1 + s0 + d1 * 768 + d2 * 64 + d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d4)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
module {
  func.func @contract_4d() attributes {hal.executable.target = #executable_target_cuda_nvptx_fb, translation_info = #translation} {
    %c12 = arith.constant 12 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 8.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.constant.load[0] : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%1) : memref<?x?x12x64xf32>{%1, %1}
    %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%1) : memref<?x?x12x64xf32>{%1, %1}
    %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<?x12x?x?xf32>{%1, %1, %1}
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %workgroup_count_z = hal.interface.workgroup.count[2] : index
    scf.for %arg0 = %workgroup_id_z to %c12 step %workgroup_count_z {
      %5 = affine.apply #map()[%workgroup_id_y]
      %6 = affine.apply #map()[%workgroup_count_y]
      scf.for %arg1 = %5 to %1 step %6 {
        %7 = affine.apply #map1()[%workgroup_id_x]
        %8 = affine.apply #map1()[%workgroup_count_x]
        scf.for %arg2 = %7 to %1 step %8 {
          %9 = affine.min #map2(%arg1)[%1]
          %10 = affine.min #map3(%arg2)[%1]
          %subview = memref.subview %4[0, %arg0, %arg1, %arg2] [%1, 1, %9, %10] [1, 1, 1, 1] : memref<?x12x?x?xf32> to memref<?x1x?x?xf32, #map4>
          %subview_1 = memref.subview %2[0, %arg1, %arg0, 0] [%1, %9, 1, 64] [1, 1, 1, 1] : memref<?x?x12x64xf32> to memref<?x?x1x64xf32, #map5>
          %subview_2 = memref.subview %3[0, %arg2, %arg0, 0] [%1, %10, 1, 64] [1, 1, 1, 1] : memref<?x?x12x64xf32> to memref<?x?x1x64xf32, #map5>
          linalg.fill {lowering_config = #config} ins(%cst_0 : f32) outs(%subview : memref<?x1x?x?xf32, #map4>)
          linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%subview_1, %subview_2 : memref<?x?x1x64xf32, #map5>, memref<?x?x1x64xf32, #map5>) outs(%subview : memref<?x1x?x?xf32, #map4>) attrs =  {lowering_config = #config} {
          ^bb0(%in: f32, %in_3: f32, %out: f32):
            %11 = arith.mulf %in, %in_3 : f32
            %12 = arith.addf %11, %out : f32
            linalg.yield %12 : f32
          }
        }
      }
    }
    return
  }
}

// Check that we are able to distribute correctly
// CHECK-LABEL: func.func @contract_4d
