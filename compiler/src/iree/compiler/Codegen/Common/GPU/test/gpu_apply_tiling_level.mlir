// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-apply-tiling-level, canonicalize, cse))" %s | FileCheck %s
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-apply-tiling-level{tiling-level=thread}, canonicalize, cse))" %s | FileCheck %s --check-prefix=THREAD
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-apply-tiling-level{tiling-level=subgroup}, canonicalize, cse))" %s | FileCheck %s --check-prefix=SUBGROUP

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#config = #iree_gpu.lowering_config<{thread = [2, 16], subgroup = [2, 16]}>
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_tensor() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x256xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x256xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x256xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [%c0, %c0], sizes = [64, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x256xf32>> -> tensor<64x256xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [%c0, %c0], sizes = [64, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x256xf32>> -> tensor<64x256xf32>
    %5 = flow.dispatch.tensor.load %2, offsets = [%c0, %c0], sizes = [64, 256], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<64x256xf32>> -> tensor<64x256xf32>
    %6 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
      } ins(%3, %4 : tensor<64x256xf32>, tensor<64x256xf32>) outs(%5 : tensor<64x256xf32>) attrs =  {lowering_config = #config} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<64x256xf32>
    flow.dispatch.tensor.store %6, %2, offsets = [%c0, %c0], sizes = [64, 256], strides = [1, 1] : tensor<64x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x256xf32>>
    return
  }
}

// Verify that no loops are generated without a reduction configuration.
// CHECK-LABEL: func.func @add_tensor
//   CHECK-NOT:   scf.for

// THREAD-LABEL: func.func @add_tensor
//       THREAD:   scf.forall ({{.*}}) = (0, 0) to (64, 256) step (2, 16)
//       THREAD:     linalg.generic {{.*}} ins(%{{.*}}: tensor<2x16xf32>, tensor<2x16xf32>)
//       THREAD:     scf.forall.in_parallel
//       THREAD:   mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>]

// SUBGROUP-LABEL: func.func @add_tensor
//       SUBGROUP:   scf.forall ({{.*}}) = (0, 0) to (64, 256) step (2, 16)
//       SUBGROUP:     linalg.generic {{.*}} ins(%{{.*}}: tensor<2x16xf32>, tensor<2x16xf32>)
//       SUBGROUP:     scf.forall.in_parallel
//       SUBGROUP:   mapping = [#gpu.warp<linear_dim_0>, #gpu.warp<linear_dim_1>]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#config = #iree_gpu.lowering_config<{thread = [0, 16]}>
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @sequential_forall_mappings() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x256xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x256xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4x256xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [%c0, %c0], sizes = [4, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4x256xf32>> -> tensor<4x256xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [%c0, %c0], sizes = [4, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4x256xf32>> -> tensor<4x256xf32>
    %5 = flow.dispatch.tensor.load %2, offsets = [%c0, %c0], sizes = [4, 256], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<4x256xf32>> -> tensor<4x256xf32>
    %6 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
      } ins(%3, %4 : tensor<4x256xf32>, tensor<4x256xf32>) outs(%5 : tensor<4x256xf32>) attrs =  {lowering_config = #config} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<4x256xf32>
    flow.dispatch.tensor.store %6, %2, offsets = [%c0, %c0], sizes = [4, 256], strides = [1, 1] : tensor<4x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x256xf32>>
    return
  }
}

// Verify that no loops are generated without a reduction configuration.
// CHECK-LABEL: func.func @sequential_forall_mappings
//   CHECK-NOT:   scf.for

// THREAD-LABEL: func.func @sequential_forall_mappings
//       THREAD:   scf.forall ({{.*}}) = (0) to (256) step (16)
//       THREAD:     linalg.generic {{.*}} ins(%{{.*}}: tensor<4x16xf32>, tensor<4x16xf32>)
//       THREAD:     scf.forall.in_parallel
//       THREAD:   mapping = [#gpu.thread<linear_dim_0>]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
func.func @matmul_transpose_b() attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorize workgroup_size = [128, 2, 1] subgroup_size = 64>} {
  %c4 = arith.constant 4 : index
  %c1280 = arith.constant 1280 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
  %5 = flow.dispatch.tensor.load %2, offsets = [%3, %4], sizes = [64, 64], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>> -> tensor<64x64xf32>
  %6 = flow.dispatch.tensor.load %0, offsets = [%3, 0], sizes = [64, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<64x1280xf16>
  %7 = flow.dispatch.tensor.load %1, offsets = [%4, 0], sizes = [64, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>> -> tensor<64x1280xf16>
  %8 = linalg.fill ins(%cst : f16) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %9 = tensor.empty() : tensor<64x1280xf16>
  %10 = tensor.empty() : tensor<64x1280xf16>
  %11 = scf.for %arg0 = %c0 to %c1280 step %c4 iter_args(%arg1 = %8) -> (tensor<64x64xf32>) {
    %extracted_slice = tensor.extract_slice %6[0, %arg0] [64, 4] [1, 1] : tensor<64x1280xf16> to tensor<64x4xf16>
    %extracted_slice_0 = tensor.extract_slice %9[0, %arg0] [64, 4] [1, 1] : tensor<64x1280xf16> to tensor<64x4xf16>
    %12 = linalg.copy {lowering_config = #iree_gpu.lowering_config<{thread = [1, 1]}>} ins(%extracted_slice : tensor<64x4xf16>) outs(%extracted_slice_0 : tensor<64x4xf16>) -> tensor<64x4xf16>
    %extracted_slice_1 = tensor.extract_slice %7[0, %arg0] [64, 4] [1, 1] : tensor<64x1280xf16> to tensor<64x4xf16>
    %extracted_slice_2 = tensor.extract_slice %10[0, %arg0] [64, 4] [1, 1] : tensor<64x1280xf16> to tensor<64x4xf16>
    %13 = linalg.copy {lowering_config = #iree_gpu.lowering_config<{thread = [1, 1]}>} ins(%extracted_slice_1 : tensor<64x4xf16>) outs(%extracted_slice_2 : tensor<64x4xf16>) -> tensor<64x4xf16>
    %14 = linalg.matmul_transpose_b {lowering_config = #iree_gpu.lowering_config<{thread = [4, 4]}>} ins(%12, %13 : tensor<64x4xf16>, tensor<64x4xf16>) outs(%arg1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    scf.yield %14 : tensor<64x64xf32>
  }
  flow.dispatch.tensor.store %11, %2, offsets = [%3, %4], sizes = [64, 64], strides = [1, 1] : tensor<64x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
  return
}

// CHECK-LABEL: func.func @matmul_transpose_b

// THREAD-LABEL: func.func @matmul_transpose_b
//       THREAD:   scf.forall ({{.*}}) in (64, 4)
//       THREAD:     linalg.copy
//       THREAD:   mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>]
//       THREAD:   scf.forall ({{.*}}) in (64, 4)
//       THREAD:     linalg.copy
//       THREAD:   mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>]
//       THREAD:   scf.forall ({{.*}}) = (0, 0) to (64, 64) step (4, 4)
//       THREAD:     linalg.matmul
//       THREAD:   mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#config = #iree_gpu.lowering_config<{reduction = [0, 8]}>
#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
func.func @reduction() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x384xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x384xf32>> -> tensor<128x384xf32>
  %empty = tensor.empty() : tensor<128xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %5 = linalg.generic {
    indexing_maps = [#map1, #map2],
    iterator_types = ["parallel", "reduction"]
    } ins(%3 : tensor<128x384xf32>) outs(%4 : tensor<128xf32>) attrs =  {lowering_config = #config} {
  ^bb0(%in: f32, %out: f32):
    %7 = arith.addf %in, %out : f32
    linalg.yield %7 : f32
  } -> tensor<128xf32>
  flow.dispatch.tensor.store %5, %1, offsets = [%c0], sizes = [128], strides = [1] : tensor<128xf32> -> !flow.dispatch.tensor<writeonly:tensor<128xf32>>
  return
}

// CHECK-LABEL: func.func @reduction
//       CHECK:   %[[FILL:.+]] = linalg.fill {{.*}} tensor<128xf32>
//       CHECK:   scf.for %{{.*}} = %c0 to %c384 step %c8 iter_args(%{{.*}} = %[[FILL]])
//       CHECK:     linalg.generic {{.*}} ins(%{{.*}} : tensor<128x8xf32>)
//       CHECK:     scf.yield

// Verify that no tiling happens in the thread case.
// THREAD-LABEL: func.func @reduction
//   THREAD-NOT:   scf.forall

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#config = #iree_gpu.lowering_config<{reduction = [0, 0, 8]}>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @matmul_fuse() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.0 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x64xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x64xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x64xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [%c0, %c0], sizes = [64, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x64xf32>> -> tensor<64x64xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [%c0, %c0], sizes = [64, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x64xf32>> -> tensor<64x64xf32>
  %5 = flow.dispatch.tensor.load %2, offsets = [%c0, %c0], sizes = [64, 64], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<64x64xf32>> -> tensor<64x64xf32>
  %empty = tensor.empty() : tensor<64x64xf32>
  %6 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]
    } ins(%3 : tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %8 = arith.addf %in, %cst : f32
    linalg.yield %8 : f32
  } -> tensor<64x64xf32>
  %7 = linalg.matmul {lowering_config = #config} ins(%6, %4 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [%c0, %c0], sizes = [64, 64], strides = [1, 1] : tensor<64x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x64xf32>>
  return
}

// CHECK-LABEL: func.func @matmul_fuse
//       CHECK:   scf.for %{{.*}} = %c0 to %c64 step %c8
//       CHECK:     %[[ELEMWISE:.+]] = linalg.generic {{.*}} ins(%{{.*}} : tensor<64x8xf32>)
//       CHECK:     %[[MM:.+]] = linalg.matmul {{.*}} ins(%[[ELEMWISE]], {{.*}} : tensor<64x8xf32>, tensor<8x64xf32>)

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#config = #iree_gpu.lowering_config<{thread = [8, 8]}>
func.func @matmul_cleanup() {
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x64xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x64xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x64xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x64xf32>> -> tensor<64x64xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x64xf32>> -> tensor<64x64xf32>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<64x64xf32>> -> tensor<64x64xf32>
  %6 = scf.for %arg0 = %c0 to %c64 step %c8 iter_args(%arg1 = %5) -> (tensor<64x64xf32>) {
    %extracted_slice = tensor.extract_slice %3[0, %arg0] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
    %extracted_slice_0 = tensor.extract_slice %4[%arg0, 0] [8, 64] [1, 1] : tensor<64x64xf32> to tensor<8x64xf32>
    %7 = linalg.matmul {lowering_config = #config} ins(%extracted_slice, %extracted_slice_0 : tensor<64x8xf32>, tensor<8x64xf32>) outs(%arg1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    scf.yield %7 : tensor<64x64xf32>
  }
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : tensor<64x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x64xf32>>
  return
}

// THREAD-LABEL: func.func @matmul_cleanup
//       THREAD:   %[[B0:.+]] = hal.interface.binding.subspan layout({{.+}}) set(0) binding(0)
//       THREAD:   %[[B1:.+]] = hal.interface.binding.subspan layout({{.+}}) set(0) binding(1)
//       THREAD:   %[[A:.+]] = flow.dispatch.tensor.load %[[B0]]
//       THREAD:   %[[B:.+]] = flow.dispatch.tensor.load %[[B1]]
//       THREAD:   scf.for %{{.*}} = %c0 to %c64 step %c8
//       THREAD:     scf.forall
//   THREAD-DAG:       %[[LHS:.+]] = tensor.extract_slice %[[A]]
//   THREAD-DAG:       %[[RHS:.+]] = tensor.extract_slice %[[B]]
//       THREAD:       %[[MM:.+]] = linalg.matmul {{.*}} ins(%[[LHS]], %[[RHS]] : tensor<8x8xf32>, tensor<8x8xf32>)

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#config = #iree_gpu.derived_thread_config
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @inferred_add_tensor()
      attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorize workgroup_size = [16, 32, 1] subgroup_size = 64, {}>} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x256xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x256xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x256xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [%c0, %c0], sizes = [64, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x256xf32>> -> tensor<64x256xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [%c0, %c0], sizes = [64, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x256xf32>> -> tensor<64x256xf32>
    %5 = flow.dispatch.tensor.load %2, offsets = [%c0, %c0], sizes = [64, 256], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<64x256xf32>> -> tensor<64x256xf32>
    %6 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
      } ins(%3, %4 : tensor<64x256xf32>, tensor<64x256xf32>) outs(%5 : tensor<64x256xf32>) attrs =  {lowering_config = #config} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<64x256xf32>
    flow.dispatch.tensor.store %6, %2, offsets = [%c0, %c0], sizes = [64, 256], strides = [1, 1] : tensor<64x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x256xf32>>
    return
  }
}

// Verify that no loops are generated without a reduction configuration.
// CHECK-LABEL: func.func @inferred_add_tensor
//   CHECK-NOT:   scf.for

// THREAD-LABEL: func.func @inferred_add_tensor
//       THREAD:   scf.forall ({{.*}}) = (0, 0) to (64, 256) step (8, 4)
//       THREAD:     linalg.generic {{.*}} ins(%{{.*}}: tensor<8x4xf32>, tensor<8x4xf32>)
//       THREAD:     scf.forall.in_parallel
//       THREAD:   mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#config = #iree_gpu.derived_thread_config
module {
  func.func @inferred_im2col()
      attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorize workgroup_size = [16, 32, 1] subgroup_size = 64, {}>} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x34x34x128xf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x128x8xf16>>
    %2 = flow.dispatch.tensor.load %0, offsets = [%c0, %c0, %c0, %c0], sizes = [2, 34, 34, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x34x34x128xf16>> -> tensor<2x34x34x128xf16>
    %3 = flow.dispatch.tensor.load %1, offsets = [%c0, %c0, %c0], sizes = [2, 128, 8], strides = [1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<2x128x8xf16>> -> tensor<2x128x8xf16>
    %4 = iree_linalg_ext.im2col {lowering_config = #config} strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3] m_offset = [0] k_offset = [0] batch_pos = [0] m_pos = [2, 3] k_pos = [1] ins(%2 : tensor<2x34x34x128xf16>) outs(%3 : tensor<2x128x8xf16>) -> tensor<2x128x8xf16>
    flow.dispatch.tensor.store %4, %1, offsets = [%c0, %c0, %c0], sizes = [2, 128, 8], strides = [1, 1, 1] : tensor<2x128x8xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x128x8xf16>>
    return
  }
}

// Verify that no loops are generated without a reduction configuration.
// CHECK-LABEL: func.func @inferred_im2col
//   CHECK-NOT:   scf.for

// THREAD-LABEL: func.func @inferred_im2col
//       THREAD:   scf.forall ({{.*}}) = (0, 0, 0) to (2, 128, 8) step (1, 1, 4)
//       THREAD:     iree_linalg_ext.im2col {{.*}} ins(%{{.*}}: tensor<1x34x34x128xf16>) outs({{.*}}: tensor<1x1x4xf16>)
//       THREAD:     scf.forall.in_parallel
//       THREAD:   mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_2>]

// -----

#config = #iree_gpu.lowering_config<{reduction = [0, 0, 8], subgroup = [2, 4]}>
#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]

module {
  func.func @tensor_multi_mma(%lhs: tensor<?x?x4xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
    %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
      lowering_config = #config
    } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
    return %0 : tensor<?x?x4xf32>
  }
}

// CHECK-LABEL: func.func @tensor_multi_mma
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<?x?x4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<?x?x4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: tensor<?x?x4xf32>

//   CHECK-DAG:   %[[MDIM:.+]] = tensor.dim %[[ACC]], %c0 : tensor<?x?x4xf32>
//   CHECK-DAG:   %[[NDIM:.+]] = tensor.dim %[[ACC]], %c1 : tensor<?x?x4xf32>
//   CHECK-DAG:   %[[KDIM:.+]] = tensor.dim %[[LHS]], %c1 : tensor<?x?x4xf16>
//       CHECK:   scf.for %[[I:.+]] = %c0 to %[[KDIM]] step %c8 iter_args(%[[INIT:.+]] = %[[ACC]])
//       CHECK:     %[[MIN:.+]] = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 8)>(%[[I]])[%[[KDIM]]]
//       CHECK:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]]
//  CHECK-SAME:       [0, %[[I]], 0] [%[[MDIM]], %[[MIN]], 4] [1, 1, 1] : tensor<?x?x4xf16> to tensor<?x?x4xf16>
//       CHECK:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]]
//  CHECK-SAME:       [%[[I]], 0, 0] [%[[MIN]], %[[NDIM]], 4] [1, 1, 1] : tensor<?x?x4xf16> to tensor<?x?x4xf16>
//       CHECK:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[INIT]]
//  CHECK-SAME:       [0, 0, 0] [%[[MDIM]], %[[NDIM]], 4] [1, 1, 1] : tensor<?x?x4xf32> to tensor<?x?x4xf32>
//       CHECK:     %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_SLICE]], %[[RHS_SLICE]], %[[ACC_SLICE]] {{.*}} lowering_config
//       CHECK:     tensor.insert_slice %[[MMA]] into %[[INIT]]
//  CHECK-SAME:       [0, 0, 0] [%[[MDIM]], %[[NDIM]], 4] [1, 1, 1]
//       CHECK:     scf.yield
//       CHECK:   return

// SUBGROUP-LABEL: func.func @tensor_multi_mma
//  SUBGROUP-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<?x?x4xf16>
//  SUBGROUP-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<?x?x4xf16>
//  SUBGROUP-SAME:   %[[ACC:[A-Za-z0-9]+]]: tensor<?x?x4xf32>

//   SUBGROUP-DAG:   %[[MDIM:.+]] = tensor.dim %[[ACC]], %c0 : tensor<?x?x4xf32>
//   SUBGROUP-DAG:   %[[NDIM:.+]] = tensor.dim %[[ACC]], %c1 : tensor<?x?x4xf32>
//   SUBGROUP-DAG:   %[[KDIM:.+]] = tensor.dim %[[LHS]], %c1 : tensor<?x?x4xf16>

//       SUBGROUP:   scf.forall (%[[IDX:.+]], %[[IDY:.+]]) = (0, 0) to (%[[MDIM]], %[[NDIM]]) step (2, 4)
//  SUBGROUP-SAME:     shared_outs(%[[INIT:.+]] = %[[ACC]])
//   SUBGROUP-DAG:     %[[MMIN:.+]] = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 2)>(%[[IDX]])[%[[MDIM]]]
//   SUBGROUP-DAG:     %[[NMIN:.+]] = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 4)>(%[[IDY]])[%[[NDIM]]]

//       SUBGROUP:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]]
//  SUBGROUP-SAME:       [%[[IDX]], 0, 0] [%[[MMIN]], %[[KDIM]], 4] [1, 1, 1] : tensor<?x?x4xf16> to tensor<?x?x4xf16>
//       SUBGROUP:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]]
//  SUBGROUP-SAME:       [0, %[[IDY]], 0] [%[[KDIM]], %[[NMIN]], 4] [1, 1, 1] : tensor<?x?x4xf16> to tensor<?x?x4xf16>
//       SUBGROUP:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[INIT]]
//  SUBGROUP-SAME:       [%[[IDX]], %[[IDY]], 0] [%[[MMIN]], %[[NMIN]], 4] [1, 1, 1] : tensor<?x?x4xf32> to tensor<?x?x4xf32>
//       SUBGROUP:     %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_SLICE]], %[[RHS_SLICE]], %[[ACC_SLICE]] {{.*}} lowering_config
//       SUBGROUP:     scf.forall.in_parallel
//       SUBGROUP:       tensor.parallel_insert_slice %[[MMA]] into %[[INIT]]
//       SUBGROUP:   return
