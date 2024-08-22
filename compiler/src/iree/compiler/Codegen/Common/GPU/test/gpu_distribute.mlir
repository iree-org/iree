// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-distribute, cse))" %s --split-input-file | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
#map2 = affine_map<(d0) -> (d0 * 4)>
#translation = #iree_codegen.translation_info<LLVMGPUVectorize workgroup_size = [64, 1, 1]>
func.func @add_tensor() attributes {translation_info = #translation} {
  %cst = arith.constant 0.000000e+00 : f32
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<233x1024xf32>
  memref.assume_alignment %0, 64 : memref<233x1024xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<233x1024xf32>
  memref.assume_alignment %1, 64 : memref<233x1024xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : memref<233x1024xf32>
  memref.assume_alignment %2, 64 : memref<233x1024xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %3 = affine.apply #map()[%workgroup_id_x]
  %subview = memref.subview %2[%workgroup_id_y, %3] [1, 256] [1, 1] : memref<233x1024xf32> to memref<1x256xf32, #map1>
  %subview_0 = memref.subview %0[%workgroup_id_y, %3] [1, 256] [1, 1] : memref<233x1024xf32> to memref<1x256xf32, #map1>
  %subview_1 = memref.subview %1[%workgroup_id_y, %3] [1, 256] [1, 1] : memref<233x1024xf32> to memref<1x256xf32, #map1>
  scf.forall (%arg0) in (%c64) {
    %4 = affine.apply #map2(%arg0)
    %subview_2 = memref.subview %subview[0, %4] [1, 4] [1, 1] : memref<1x256xf32, #map1> to memref<1x4xf32, #map1>
    %5 = vector.transfer_read %subview_0[%c0, %4], %cst {in_bounds = [true]} : memref<1x256xf32, #map1>, vector<4xf32>
    %6 = vector.transfer_read %subview_1[%c0, %4], %cst {in_bounds = [true]} : memref<1x256xf32, #map1>, vector<4xf32>
    %7 = arith.addf %5, %6 : vector<4xf32>
    vector.transfer_write %7, %subview_2[%c0, %c0] {in_bounds = [true]} : vector<4xf32>, memref<1x4xf32, #map1>
  } {mapping = [#gpu.thread<x>]}
  return
}

//         CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0 * 4)>
//   CHECK-LABEL: func.func @add_tensor
//         CHECK:   %[[C0:.*]] = arith.constant 0 : index
//         CHECK:   %[[TX:.*]] = gpu.thread_id  x
//         CHECK:   %[[OFF:.*]] = affine.apply #[[$MAP]](%[[TX]])
//         CHECK:   %[[S:.*]] = memref.subview %{{.*}}[0, %[[OFF]]] [1, 4] [1, 1] : memref<1x256xf32, #{{.*}}> to memref<1x4xf32, #{{.*}}>
//         CHECK:   %[[A:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[OFF]]], %{{.*}} {in_bounds = [true]} : memref<1x256xf32, #{{.*}}>, vector<4xf32>
//         CHECK:   %[[B:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[OFF]]], %{{.*}} {in_bounds = [true]} : memref<1x256xf32, #{{.*}}>, vector<4xf32>
//         CHECK:   %[[C:.*]] = arith.addf %[[A]], %[[B]] : vector<4xf32>
//         CHECK:   vector.transfer_write %[[C]], %[[S]][%[[C0]], %[[C0]]] {in_bounds = [true]} : vector<4xf32>, memref<1x4xf32, #{{.*}}>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
#map2 = affine_map<(d0) -> (d0 * 4)>
#translation = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 1, 1]>
func.func @add_tensor_lane_id() attributes {translation_info = #translation} {
  %cst = arith.constant 0.000000e+00 : f32
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<233x1024xf32>
  memref.assume_alignment %0, 64 : memref<233x1024xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<233x1024xf32>
  memref.assume_alignment %1, 64 : memref<233x1024xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : memref<233x1024xf32>
  memref.assume_alignment %2, 64 : memref<233x1024xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %3 = affine.apply #map()[%workgroup_id_x]
  %subview = memref.subview %2[%workgroup_id_y, %3] [1, 256] [1, 1] : memref<233x1024xf32> to memref<1x256xf32, #map1>
  %subview_0 = memref.subview %0[%workgroup_id_y, %3] [1, 256] [1, 1] : memref<233x1024xf32> to memref<1x256xf32, #map1>
  %subview_1 = memref.subview %1[%workgroup_id_y, %3] [1, 256] [1, 1] : memref<233x1024xf32> to memref<1x256xf32, #map1>
  scf.forall (%arg0) in (%c64) {
    %4 = affine.apply #map2(%arg0)
    %subview_2 = memref.subview %subview[0, %4] [1, 4] [1, 1] : memref<1x256xf32, #map1> to memref<1x4xf32, #map1>
    %5 = vector.transfer_read %subview_0[%c0, %4], %cst {in_bounds = [true]} : memref<1x256xf32, #map1>, vector<4xf32>
    %6 = vector.transfer_read %subview_1[%c0, %4], %cst {in_bounds = [true]} : memref<1x256xf32, #map1>, vector<4xf32>
    %7 = arith.addf %5, %6 : vector<4xf32>
    vector.transfer_write %7, %subview_2[%c0, %c0] {in_bounds = [true]} : vector<4xf32>, memref<1x4xf32, #map1>
  } {mapping = [#iree_gpu.lane_id<0>]}
  return
}

//         CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0 * 4)>
//   CHECK-LABEL: func.func @add_tensor_lane_id
//         CHECK:   %[[C0:.*]] = arith.constant 0 : index
//         CHECK:   %[[TX:.*]] = gpu.lane_id
//         CHECK:   %[[OFF:.*]] = affine.apply #[[$MAP]](%[[TX]])
//         CHECK:   %[[S:.*]] = memref.subview %{{.*}}[0, %[[OFF]]] [1, 4] [1, 1] : memref<1x256xf32, #{{.*}}> to memref<1x4xf32, #{{.*}}>
//         CHECK:   %[[A:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[OFF]]], %{{.*}} {in_bounds = [true]} : memref<1x256xf32, #{{.*}}>, vector<4xf32>
//         CHECK:   %[[B:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[OFF]]], %{{.*}} {in_bounds = [true]} : memref<1x256xf32, #{{.*}}>, vector<4xf32>
//         CHECK:   %[[C:.*]] = arith.addf %[[A]], %[[B]] : vector<4xf32>
//         CHECK:   vector.transfer_write %[[C]], %[[S]][%[[C0]], %[[C0]]] {in_bounds = [true]} : vector<4xf32>, memref<1x4xf32, #{{.*}}>
