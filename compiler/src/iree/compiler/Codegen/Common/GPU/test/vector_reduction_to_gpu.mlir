// RUN: iree-opt --split-input-file --iree-gpu-test-target=sm_60 --pass-pipeline='builtin.module(func.func(iree-codegen-vector-reduction-to-gpu, cse))' %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx940 --pass-pipeline='builtin.module(func.func(iree-codegen-vector-reduction-to-gpu, cse))' %s | FileCheck %s --check-prefix=CDNA3

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<()[s0, s1] -> (s1 * 2 + s0 floordiv 32)>
#translation_info = #iree_codegen.translation_info<None workgroup_size = [32, 1, 1] subgroup_size = 32>
module {
  func.func @simple_reduce() attributes {translation_info = #translation_info} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense<3.840000e+02> : vector<1xf32>
    %c32 = arith.constant 32 : index
    %c384 = arith.constant 384 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<128x384xf32>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<128xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %thread_id_x = gpu.thread_id  x
    %2 = affine.apply #map()[%thread_id_x, %workgroup_id_x]
    %3 = scf.for %arg0 = %c0 to %c384 step %c32 iter_args(%arg1 = %cst) -> (vector<1xf32>) {
      %5 = vector.transfer_read %0[%2, %arg0], %cst_0 {in_bounds = [true]} : memref<128x384xf32>, vector<32xf32>
      %6 = vector.broadcast %5 : vector<32xf32> to vector<1x32xf32>
      %7 = vector.multi_reduction <add>, %6, %arg1 [1] : vector<1x32xf32> to vector<1xf32>
      scf.yield %7 : vector<1xf32>
    }
    %4 = arith.divf %3, %cst_1 : vector<1xf32>
    vector.transfer_write %4, %1[%2] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
    return
  }
}

// CHECK-LABEL: func.func @simple_reduce()
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : i32
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : i32
//   CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : i32
//   CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : i32
//   CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : i32
//   CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : i32
//   CHECK-DAG:   %[[C32I:.*]] = arith.constant 32 : index
//   CHECK-DAG:   %[[TID:.*]] = gpu.thread_id  x
//   CHECK-DAG:   %[[VCST:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
//       CHECK:   %[[F:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[V0:.*]] = %[[VCST]]) -> (vector<1xf32>) {
//   CHECK-DAG:     %[[E:.*]] = vector.extractelement %[[V0]][%[[C0]] : index] : vector<1xf32>
//   CHECK-DAG:     %[[ID:.*]] = affine.apply
//   CHECK-DAG:     %[[V1:.*]] = vector.transfer_read %{{.*}}[%{{.*}}, %[[ID]]], %{{.*}} {in_bounds = [true]} : memref<128x384xf32>, vector<1xf32>
//       CHECK:     %[[S:.*]] = vector.extract %[[V1]][0] : f32 from vector<1xf32>
//       CHECK:     %[[S0:.*]], %{{.*}} = gpu.shuffle  xor %[[S]], %[[C1]], %[[C32]] : f32
//       CHECK:     %[[S1:.*]] = arith.addf %[[S]], %[[S0]] : f32
//       CHECK:     %[[S2:.*]], %{{.*}} = gpu.shuffle  xor %[[S1]], %[[C2]], %[[C32]] : f32
//       CHECK:     %[[S3:.*]] = arith.addf %[[S1]], %[[S2]] : f32
//       CHECK:     %[[S4:.*]], %{{.*}} = gpu.shuffle  xor %[[S3]], %[[C4]], %[[C32]] : f32
//       CHECK:     %[[S5:.*]] = arith.addf %[[S3]], %[[S4]] : f32
//       CHECK:     %[[S6:.*]], %{{.*}} = gpu.shuffle  xor %[[S5]], %[[C8]], %[[C32]] : f32
//       CHECK:     %[[S7:.*]] = arith.addf %[[S5]], %[[S6]] : f32
//       CHECK:     %[[S8:.*]], %{{.*}} = gpu.shuffle  xor %[[S7]], %[[C16]], %[[C32]] : f32
//       CHECK:     %[[S9:.*]] = arith.addf %[[S7]], %[[S8]] : f32
//       CHECK:     %[[S10:.*]] = arith.addf %[[S9]], %[[E]] : f32
//       CHECK:     %[[B:.*]] = vector.broadcast %[[S10]] : f32 to vector<1xf32>
//       CHECK:     scf.yield %[[B]] : vector<1xf32>
//       CHECK:   }
//       CHECK:   %[[DIV:.*]] = arith.divf %[[F]], %{{.*}} : vector<1xf32>
//       CHECK:   %[[CMP:.*]] = arith.cmpi eq, %[[TID]], %[[C0]] : index
//       CHECK:   scf.if %[[CMP]] {
//       CHECK:     vector.transfer_write %[[DIV]], {{.*}} : vector<1xf32>, memref<128xf32>
//       CHECK:   }
//       CHECK:   return

// -----

// Make sure memref.load from uniform buffers are hoisted out as uniform code.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<uniform_buffer>
]>
#translation_info = #iree_codegen.translation_info<None workgroup_size = [32, 1, 1] subgroup_size = 32>
#map = affine_map<()[s0, s1] -> (s1 * 2 + s0 floordiv 32)>
module {
  func.func @reduce_uniform_buffer_offset() attributes {translation_info = #translation_info} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense<3.840000e+02> : vector<1xf32>
    %c32 = arith.constant 32 : index
    %c384 = arith.constant 384 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) offset(%c0) : memref<1xvector<4xi32>, #hal.descriptor_type<uniform_buffer>>
    %1 = memref.load %0[%c0] : memref<1xvector<4xi32>, #hal.descriptor_type<uniform_buffer>>
    %2 = vector.extractelement %1[%c0 : index] : vector<4xi32>
    %3 = vector.extractelement %1[%c1 : index] : vector<4xi32>
    %4 = arith.index_castui %2 : i32 to index
    %5 = arith.index_castui %3 : i32 to index
    %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%4) : memref<128x384xf32>
    %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%5) : memref<128xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %thread_id_x = gpu.thread_id  x
    %8 = affine.apply #map()[%thread_id_x, %workgroup_id_x]
    %9 = scf.for %arg0 = %c0 to %c384 step %c32 iter_args(%arg1 = %cst) -> (vector<1xf32>) {
      %11 = vector.transfer_read %6[%8, %arg0], %cst_0 {in_bounds = [true]} : memref<128x384xf32>, vector<32xf32>
      %12 = vector.broadcast %11 : vector<32xf32> to vector<1x32xf32>
      %13 = vector.multi_reduction <add>, %12, %arg1 [1] : vector<1x32xf32> to vector<1xf32>
      scf.yield %13 : vector<1xf32>
    }
    %10 = arith.divf %9, %cst_1 : vector<1xf32>
    vector.transfer_write %10, %7[%8] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
    return
  }
}

//   CHECK-LABEL: func.func @reduce_uniform_buffer_offset()
//     CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//     CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//         CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//         CHECK:   %[[LOAD:.+]] = memref.load %[[SUBSPAN]][%[[C0]]]
//         CHECK:   %[[EXT0:.+]] = vector.extractelement %[[LOAD]][%[[C0]] : index] : vector<4xi32>
//         CHECK:   %[[EXT1:.+]] = vector.extractelement %[[LOAD]][%[[C1]] : index] : vector<4xi32>
//         CHECK:   %[[OFFSET0:.+]] = arith.index_castui %[[EXT0]] : i32 to index
//         CHECK:   %[[OFFSET1:.+]] = arith.index_castui %[[EXT1]] : i32 to index
//         CHECK:   hal.interface.binding.subspan layout({{.+}}) binding(0) alignment(64) offset(%[[OFFSET0]])
//         CHECK:   hal.interface.binding.subspan layout({{.+}}) binding(1) alignment(64) offset(%[[OFFSET1]])
//         CHECK:   scf.for
// CHECK-COUNT-5:     gpu.shuffle
//         CHECK:     arith.addf
//         CHECK:     scf.yield

// -----

// Make sure memref.load from readonly storage buffers are hoisted out as uniform code.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<()[s0, s1] -> (s1 * 2 + s0 floordiv 32)>
#translation_info = #iree_codegen.translation_info<None workgroup_size = [32, 1, 1] subgroup_size = 32>
module {
  func.func @reduce_storage_buffer_offset() attributes {translation_info = #translation_info} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense<3.840000e+02> : vector<1xf32>
    %c32 = arith.constant 32 : index
    %c384 = arith.constant 384 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : memref<1xvector<4xi32>, #hal.descriptor_type<storage_buffer>>
    %1 = memref.load %0[%c0] : memref<1xvector<4xi32>, #hal.descriptor_type<storage_buffer>>
    %2 = vector.extractelement %1[%c0 : index] : vector<4xi32>
    %3 = vector.extractelement %1[%c1 : index] : vector<4xi32>
    %4 = arith.index_castui %2 : i32 to index
    %5 = arith.index_castui %3 : i32 to index
    %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%4) : memref<128x384xf32>
    %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%5) : memref<128xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %thread_id_x = gpu.thread_id  x
    %8 = affine.apply #map()[%thread_id_x, %workgroup_id_x]
    %9 = scf.for %arg0 = %c0 to %c384 step %c32 iter_args(%arg1 = %cst) -> (vector<1xf32>) {
      %11 = vector.transfer_read %6[%8, %arg0], %cst_0 {in_bounds = [true]} : memref<128x384xf32>, vector<32xf32>
      %12 = vector.broadcast %11 : vector<32xf32> to vector<1x32xf32>
      %13 = vector.multi_reduction <add>, %12, %arg1 [1] : vector<1x32xf32> to vector<1xf32>
      scf.yield %13 : vector<1xf32>
    }
    %10 = arith.divf %9, %cst_1 : vector<1xf32>
    vector.transfer_write %10, %7[%8] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
    return
  }
}

//   CHECK-LABEL: func.func @reduce_storage_buffer_offset()
//     CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//     CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//         CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//         CHECK:   %[[LOAD:.+]] = memref.load %[[SUBSPAN]][%[[C0]]]
//         CHECK:   %[[EXT0:.+]] = vector.extractelement %[[LOAD]][%[[C0]] : index] : vector<4xi32>
//         CHECK:   %[[EXT1:.+]] = vector.extractelement %[[LOAD]][%[[C1]] : index] : vector<4xi32>
//         CHECK:   %[[OFFSET0:.+]] = arith.index_castui %[[EXT0]] : i32 to index
//         CHECK:   %[[OFFSET1:.+]] = arith.index_castui %[[EXT1]] : i32 to index
//         CHECK:   hal.interface.binding.subspan layout({{.+}}) binding(0) alignment(64) offset(%[[OFFSET0]])
//         CHECK:   hal.interface.binding.subspan layout({{.+}}) binding(1) alignment(64) offset(%[[OFFSET1]])
//         CHECK:   scf.for
// CHECK-COUNT-5:     gpu.shuffle
//         CHECK:     arith.addf
//         CHECK:     scf.yield

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation_info = #iree_codegen.translation_info<None workgroup_size = [32, 1, 1] subgroup_size = 32>
module {
  func.func @shared_memory_copy() attributes {translation_info = #translation_info} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32 = arith.constant 32 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<128x32xf32>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<128x32xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32, #gpu.address_space<workgroup>>
    %2 = vector.transfer_read %0[%workgroup_id_x, %c0], %cst_0 {in_bounds = [true]} : memref<128x32xf32>, vector<32xf32>
    vector.transfer_write %2, %alloc[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, #gpu.address_space<workgroup>>
    gpu.barrier
    %3 = vector.transfer_read %alloc[%c0], %cst_0 {in_bounds = [true]} : memref<32xf32, #gpu.address_space<workgroup>>, vector<32xf32>
    vector.transfer_write %3, %1[%workgroup_id_x, %c0] {in_bounds = [true]} : vector<32xf32>, memref<128x32xf32>
    return
  }
}

// CHECK-LABEL: func.func @shared_memory_copy()
//       CHECK:   %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<32xf32, #gpu.address_space<workgroup>>
//       CHECK:   vector.transfer_read {{.*}} : memref<128x32xf32>, vector<1xf32>
//       CHECK:   vector.transfer_write {{.*}} %[[ALLOC]]{{.*}} : vector<1xf32>, memref<32xf32, #gpu.address_space<workgroup>>
//       CHECK:   gpu.barrier
//       CHECK:   vector.transfer_read %[[ALLOC]]{{.*}} : memref<32xf32, #gpu.address_space<workgroup>>, vector<1xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<1xf32>, memref<128x32xf32>
//       CHECK:   return

// -----

// Check that we multi-row matvec gets distributed across subgroup threads.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation_info = #iree_codegen.translation_info<None workgroup_size = [64, 1, 1] subgroup_size = 64>
#map = affine_map<()[s0] -> (s0 * 4)>
#map1 = affine_map<(d0, d1) -> (0, d1)>
module {
  func.func @multirow() attributes {translation_info = #translation_info} {
    %cst = arith.constant dense<0.000000e+00> : vector<4x512xf16>
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
    %c4096 = arith.constant 4096 : index
    %c512 = arith.constant 512 : index
    %cst_1 = arith.constant 0.000000e+00 : f16
    %thread_id_x = gpu.thread_id  x
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<1x4096xf16, #hal.descriptor_type<storage_buffer>>
    memref.assume_alignment %0, 64 : memref<1x4096xf16, #hal.descriptor_type<storage_buffer>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<32000x4096xf16, #hal.descriptor_type<storage_buffer>>
    memref.assume_alignment %1, 64 : memref<32000x4096xf16, #hal.descriptor_type<storage_buffer>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : memref<1x32000xf16, #hal.descriptor_type<storage_buffer>>
    memref.assume_alignment %2, 64 : memref<1x32000xf16, #hal.descriptor_type<storage_buffer>>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %3 = affine.apply #map()[%workgroup_id_x]
    %4 = scf.for %arg0 = %c0 to %c4096 step %c512 iter_args(%arg1 = %cst) -> (vector<4x512xf16>) {
      %8 = vector.transfer_read %0[%c0, %arg0], %cst_1 {in_bounds = [true, true], permutation_map = #map1} : memref<1x4096xf16, #hal.descriptor_type<storage_buffer>>, vector<4x512xf16>
      %9 = vector.transfer_read %1[%3, %arg0], %cst_1 {in_bounds = [true, true]} : memref<32000x4096xf16, #hal.descriptor_type<storage_buffer>>, vector<4x512xf16>
      %10 = arith.mulf %8, %9 : vector<4x512xf16>
      %11 = arith.addf %arg1, %10 : vector<4x512xf16>
      scf.yield %11 : vector<4x512xf16>
    }
    %5 = vector.broadcast %4 : vector<4x512xf16> to vector<1x4x512xf16>
    %6 = vector.multi_reduction <add>, %5, %cst_0 [2] : vector<1x4x512xf16> to vector<1x4xf16>
    %7 = vector.extract %6[0] : vector<4xf16> from vector<1x4xf16>
    vector.transfer_write %7, %2[%c0, %3] {in_bounds = [true]} : vector<4xf16>, memref<1x32000xf16, #hal.descriptor_type<storage_buffer>>
    return
  }
}

// CDNA3-LABEL: func.func @multirow()
//       CDNA3:   scf.for {{.*}} -> (vector<4x8xf16>) {
//       CDNA3:     vector.transfer_read {{.*}} : memref<32000x4096xf16, #hal.descriptor_type<storage_buffer>>, vector<4x8xf16>
//       CDNA3:     vector.transfer_read {{.*}} : memref<1x4096xf16, #hal.descriptor_type<storage_buffer>>, vector<4x8xf16>
//       CDNA3:     arith.mulf %{{.*}}, %{{.*}} : vector<4x8xf16>
//       CDNA3:     arith.addf %{{.*}}, %{{.*}} : vector<4x8xf16>
//       CDNA3:   }
// CDNA3-COUNT-12: gpu.shuffle xor
//       CDNA3:   scf.if {{.*}} {
//       CDNA3:     vector.transfer_write {{.*}} : vector<4xf16>, memref<1x32000xf16, #hal.descriptor_type<storage_buffer>>
//       CDNA3:   }
//  CDNA3-NEXT:   return

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation_info = #iree_codegen.translation_info<None workgroup_size = [32, 1, 1] subgroup_size = 32>
module {
  func.func @simple_nd_write() attributes {translation_info = #translation_info} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<4x1024xf32>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<4x1024xf32>
    %2 = vector.transfer_read %0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x1024xf32>, vector<4x1024xf32>
    vector.transfer_write %2, %1[%c0, %c0] {in_bounds = [true, true]} : vector<4x1024xf32>, memref<4x1024xf32>
    return
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 * 128)>

// CHECK-LABEL: func @simple_nd_write(
//       CHECK:   %[[RD:.+]] = vector.transfer_read {{.*}} vector<1x128xf32>
//       CHECK:   %[[IDS:.+]]:2 = affine.delinearize_index %{{.*}} into (%c4, %c8) : index, index
//       CHECK:   %[[INNER_ID:.+]] = affine.apply #[[$MAP]]()[%[[IDS]]#1]
//       CHECK:   vector.transfer_write %[[RD]], %{{.*}}[%[[IDS]]#0, %[[INNER_ID]]] {{.*}} : vector<1x128xf32>
