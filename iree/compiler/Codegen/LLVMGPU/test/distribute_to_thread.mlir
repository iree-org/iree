// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(builtin.module(builtin.func(iree-llvmgpu-tile-and-distribute))))' %s | IreeFileCheck %s

#config = #iree_codegen.lowering.config<tile_sizes = [[2, 256, 4]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulSimt", workload_per_wg = [256, 2]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#map0 = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<()[s0] -> (s0 * 256)>
#map2 = affine_map<(d0) -> (2, -d0 + 1024)>
#map3 = affine_map<(d0) -> (256, -d0 + 1024)>
#map4 = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
hal.executable private @dot_dispatch_0  {
hal.executable.variant @cuda, target = #executable_target_cuda_nvptx_fb {
  hal.executable.entry_point @dot_dispatch_0 interface(@io) {
    translation.info = #translation,
    workgroup_size = [64 : index, 1 : index, 1 : index]
  }
  builtin.module {
    builtin.func @dot_dispatch_0() {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %0 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(0) : memref<1024x1024xf32>
      %1 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(1) : memref<1024x1024xf32>
      %2 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(2) : memref<1024x1024xf32>
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
          linalg.fill(%cst, %11) {lowering.config = #config}
              : f32, memref<2x256xf32, #map4>
          linalg.matmul {lowering.config = #config}
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
//         CHECK:  memref.global "private" @{{.*}} : memref<4x256xf32, 3>
//         CHECK:  memref.global "private" @{{.*}} : memref<2x4xf32, 3>
//     CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
//     CHECK-DAG:  %[[C2:.+]] = arith.constant 2 : index
//     CHECK-DAG:  %[[C4:.+]] = arith.constant 4 : index
//     CHECK-DAG:  %[[C256:.+]] = arith.constant 256 : index
//     CHECK-DAG:  %[[C1024:.+]] = arith.constant 1024 : index
//         CHECK:  scf.for %[[K:.+]] = %[[C0]] to %[[C1024]] step %[[C4]] {
//         CHECK:    gpu.barrier
//         CHECK:    linalg.copy(%{{.*}}, %{{.*}}) {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<2x4xf32, #{{.*}}>, memref<2x4xf32, #{{.*}}, 3>
//     CHECK-NOT:    gpu.barrier
//         CHECK:    linalg.copy(%{{.*}}, %{{.*}}) {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<4x256xf32, #{{.*}}>, memref<4x256xf32, #{{.*}}, 3>
//         CHECK:    gpu.barrier
//         CHECK:    scf.for %[[IND0:.+]] = %{{.*}} to %[[C2]] step %[[C2]] {
//         CHECK:      scf.for %[[IND1:.+]] = %{{.*}} to %[[C256]] step %[[C256]] {
//     CHECK-DAG:        %[[A:.+]] = memref.subview %17[%[[IND0]], 0] [2, 4] [1, 1] : memref<2x4xf32, #{{.*}}, 3> to memref<2x4xf32, #{{.*}}, 3>
//     CHECK-DAG:        %[[B:.+]] = memref.subview %18[0, %[[IND1]]] [4, 4] [1, 1] : memref<4x256xf32, #{{.*}}, 3> to memref<4x4xf32, #{{.*}}, 3>
//     CHECK-DAG:        %[[C:.+]] = memref.subview %11[%[[IND0]], %[[IND1]]] [2, 4] [1, 1] : memref<2x256xf32, #{{.*}}> to memref<2x4xf32, #{{.*}}>
//         CHECK:        linalg.matmul {__internal_linalg_transform__ = "vectorize", {{.*}}} ins(%[[A]], %[[B]] : memref<2x4xf32, #{{.*}}, 3>, memref<4x4xf32, #{{.*}}, 3>) outs(%[[C]] : memref<2x4xf32, #{{.*}}>)
//         CHECK:    }
//         CHECK:  }

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [[]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"LLVMGPUVectorize", workload_per_wg = []>
// Pure reducion case, skip tiling.
hal.executable @reduction_dispatch {
hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @predict_dispatch_153 interface(@io) {
      translation.info = #translation,
      workgroup_size = [1: index, 1: index, 1: index]
    }
    builtin.module  {
      builtin.func @predict_dispatch_153() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0x7FC00000 : f32
        %cst_0 = arith.constant 0xFF800000 : f32
        %0 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(0) : memref<1000xf32>
        %1 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(1) : memref<f32>
        linalg.fill(%cst_0, %1) {lowering.config = #config}  : f32, memref<f32>
        linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%0 : memref<1000xf32>) outs(%1 : memref<f32>) attrs = {lowering.config = #config} {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %2 = arith.cmpf ogt, %arg0, %arg1 : f32
          %3 = select %2, %arg0, %arg1 : f32
          %4 = arith.cmpf uno, %arg0, %arg1 : f32
          %5 = select %4, %cst, %3 : f32
          linalg.yield %5 : f32
        }
        return
      }
    }
  }
}

//      CHECK: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[]{{\]}}, native_vector_size = []>
//      CHECK: hal.executable public @reduction_dispatch
//      CHECK: linalg.fill
// CHECK-SAME:     lowering.config = #[[CONFIG]]
//      CHECK: linalg.generic
// CHECK-SAME:     ins(%{{.*}} : memref<1000xf32>) outs(%{{.*}} : memref<f32>)
// CHECK-SAME:     lowering.config = #[[CONFIG]]
