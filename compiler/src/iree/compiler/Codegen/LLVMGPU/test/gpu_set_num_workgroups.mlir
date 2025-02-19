// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-codegen-llvmgpu-configuration-pipeline)" \
// RUN:   --iree-gpu-test-target=sm_60 %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-codegen-llvmgpu-configuration-pipeline)" \
// RUN:   --iree-gpu-test-target=sm_80 %s | FileCheck %s --check-prefix=SM80

// Transform dialect attributes are tested separately.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0) -> (d0)>
func.func @add_dispatch_0() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<16384xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<16384xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<16384xf32>>
  %3 = tensor.empty() : tensor<16384xf32>
  %4 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [16384], strides = [1] : !flow.dispatch.tensor<readonly:tensor<16384xf32>> -> tensor<16384xf32>
  %5 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [16384], strides = [1] : !flow.dispatch.tensor<readonly:tensor<16384xf32>> -> tensor<16384xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16384xf32>, tensor<16384xf32>) outs(%3 : tensor<16384xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %7 = arith.addf %in, %in_0 : f32
    linalg.yield %7 : f32
  } -> tensor<16384xf32>
  flow.dispatch.tensor.store %6, %2, offsets = [0], sizes = [16384], strides = [1] : tensor<16384xf32> -> !flow.dispatch.tensor<writeonly:tensor<16384xf32>>
  return
}

//      CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>
//      CHECK: func.func @add_dispatch_0
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #iree_gpu.lowering_config<{thread = [4], workgroup = [128]}>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @dot_dispatch_1() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<2x3xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<3x4xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<2x4xf32>
  linalg.fill ins(%cst : f32) outs(%2 : memref<2x4xf32>)
  linalg.matmul ins(%0, %1 : memref<2x3xf32>, memref<3x4xf32>) outs(%2 : memref<2x4xf32>)
  return
}

//      CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [2, 4, 1] subgroup_size = 32, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>
//      CHECK: func.func @dot_dispatch_1
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.fill
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #iree_gpu.lowering_config<{reduction = [0, 0, 4], thread = [2, 1, 0], workgroup = [4, 2, 1]}>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @unaligned_k() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<128x258xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<258x64xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<128x64xf32>
  linalg.fill ins(%cst : f32) outs(%2 : memref<128x64xf32>)
  linalg.matmul ins(%0, %1 : memref<128x258xf32>, memref<258x64xf32>) outs(%2 : memref<128x64xf32>)
  return
}

//      CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 8, 1] subgroup_size = 32, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>
//      CHECK: func.func @unaligned_k
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.fill
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #iree_gpu.lowering_config<{reduction = [0, 0, 2], thread = [1, 16, 0], workgroup = [32, 128, 1]}>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
func.func @predict_dispatch_153() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0x7FC00000 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<1000xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<f32>
  linalg.fill ins(%cst_0 : f32) outs(%1 : memref<f32>)
  linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%0 : memref<1000xf32>) outs(%1 : memref<f32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.cmpf ogt, %in, %out : f32
    %3 = arith.select %2, %in, %out : f32
    %4 = arith.cmpf uno, %in, %out : f32
    %5 = arith.select %4, %cst, %3 : f32
    linalg.yield %5 : f32
  }
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUDistribute workgroup_size = [1, 1, 1]>
//      CHECK: func.func @predict_dispatch_153()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.fill
//      CHECK: linalg.generic
// CHECK-SAME:   lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @reduction_aligned2() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x128x384xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x384xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4, 128, 384], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x128x384xf32>> -> tensor<4x128x384xf32>
  %3 = tensor.empty() : tensor<128x384xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<128x384xf32>) -> tensor<128x384xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<4x128x384xf32>) outs(%4 : tensor<128x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<128x384xf32>
  flow.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : tensor<128x384xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x384xf32>>
  return
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>
//      CHECK: func.func @reduction_aligned2()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.fill
//      CHECK: linalg.generic
// CHECK-SAME:   lowering_config = #iree_gpu.lowering_config
// CHECK-SAME:     reduction = [0, 0, 4]
// CHECK-SAME:     thread = [1, 4, 0]
// CHECK-SAME:     workgroup = [1, 128, 0]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @copy_as_generic() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<?x?xi32>{%0, %1}
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<?x?xi32>{%0, %1}
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : memref<?x?xi32>) outs(%3 : memref<?x?xi32>) {
  ^bb0(%in: i32, %out: i32):
    linalg.yield %in : i32
  }
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUDistribute workgroup_size = [64, 1, 1] subgroup_size = 32>
//      CHECK: func.func @copy_as_generic()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @static_1d_fft_stage2() {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant dense<[1.000000e+00, 6.12323426E-17]> : tensor<2xf32>
  %cst_0 = arith.constant dense<[-0.000000e+00, -1.000000e+00]> : tensor<2xf32>
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
  %3 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
  %4:2 = iree_linalg_ext.fft {__internal_linalg_transform__ = "workgroup"} ins(%c2, %cst, %cst_0 : index, tensor<2xf32>, tensor<2xf32>) outs(%2, %3 : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
  flow.dispatch.tensor.store %4#0, %0, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:tensor<32xf32>>
  flow.dispatch.tensor.store %4#1, %1, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:tensor<32xf32>>
  return
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUDistribute workgroup_size = [32, 1, 1]>
//       CHECK: func.func @static_1d_fft_stage2()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: iree_linalg_ext.fft
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @static_3d_fft_stage3() {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant dense<[1.000000e+00, 0.707106769, 6.12323426E-17, -0.707106769]> : tensor<4xf32>
  %cst_0 = arith.constant dense<[-0.000000e+00, -0.707106769, -1.000000e+00, -0.707106769]> : tensor<4xf32>
  %0 = bufferization.to_memref %cst_0 : tensor<4xf32> to memref<4xf32>
  %1 = bufferization.to_memref %cst : tensor<4xf32> to memref<4xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<64x128x32xf32>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<64x128x32xf32>
  iree_linalg_ext.fft {__internal_linalg_transform__ = "workgroup"} ins(%c3, %1, %0 : index, memref<4xf32>, memref<4xf32>) outs(%2, %3 : memref<64x128x32xf32>, memref<64x128x32xf32>)
  return
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1, 8]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUDistribute workgroup_size = [32, 1, 1]>
//       CHECK: func.func @static_3d_fft_stage3()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: iree_linalg_ext.fft
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[32, 128, 64]]>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [16, 8, 1], {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @_lowering_config_test_dispatch_1() {
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>> -> tensor<256x1024xf32>
  %5 = tensor.empty() : tensor<128x1024xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  %7 = linalg.matmul {__internal_linalg_transform__ = "workgroup", compilation_info = #compilation} ins(%3, %4 : tensor<128x256xf32>, tensor<256x1024xf32>) outs(%6 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 1024], strides = [1, 1] : tensor<128x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 128, 64]{{\]}}
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [16, 8, 1], {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
//      CHECK: func.func @_lowering_config_test_dispatch_1()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.fill
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @sort_op() {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c2304000 = arith.constant 2304000 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(32) : !flow.dispatch.tensor<readonly:tensor<1x576000xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(32) : !flow.dispatch.tensor<readonly:tensor<1x576000xi32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(32) : !flow.dispatch.tensor<writeonly:tensor<1x576000xf32>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(32) offset(%c2304000) : !flow.dispatch.tensor<writeonly:tensor<1x576000xi32>>
  %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 576000], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x576000xf32>> -> tensor<1x576000xf32>
  %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1, 576000], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x576000xi32>> -> tensor<1x576000xi32>
  %6:2 = iree_linalg_ext.sort dimension(1) outs(%4, %5 : tensor<1x576000xf32>, tensor<1x576000xi32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: i32, %arg3: i32):
    %7 = arith.cmpf ogt, %arg0, %arg1 : f32
    iree_linalg_ext.yield %7 : i1
  } -> tensor<1x576000xf32>, tensor<1x576000xi32>
  flow.dispatch.tensor.store %6#0, %2, offsets = [0, 0], sizes = [1, 576000], strides = [1, 1] : tensor<1x576000xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x576000xf32>>
  flow.dispatch.tensor.store %6#1, %3, offsets = [0, 0], sizes = [1, 576000], strides = [1, 1] : tensor<1x576000xi32> -> !flow.dispatch.tensor<writeonly:tensor<1x576000xi32>>
  return
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUDistribute workgroup_size = [64, 1, 1]>
//       CHECK: func.func @sort_op()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: iree_linalg_ext.sort
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_config_sm35() {
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>> -> tensor<256x1024xf32>
  %5 = tensor.empty() : tensor<128x1024xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<128x256xf32>, tensor<256x1024xf32>) outs(%6 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 1024], strides = [1, 1] : tensor<128x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
  return
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 8, 1] subgroup_size = 32, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>
//      CHECK: func.func @matmul_config_sm35()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_config_sm80() {
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>> -> tensor<256x1024xf32>
  %5 = tensor.empty() : tensor<128x1024xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<128x256xf32>, tensor<256x1024xf32>) outs(%6 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 1024], strides = [1, 1] : tensor<128x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
  return
}

//  SM80-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCoreMmaSync workgroup_size = [64, 2, 1] subgroup_size = 32
//      SM80: func.func @matmul_config_sm80()
// SM80-SAME:     translation_info = #[[TRANSLATION]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_config_sm86() {
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>> -> tensor<256x1024xf32>
  %5 = tensor.empty() : tensor<128x1024xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<128x256xf32>, tensor<256x1024xf32>) outs(%6 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 1024], strides = [1, 1] : tensor<128x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
  return
}

//  SM80-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCoreMmaSync workgroup_size = [64, 2, 1] subgroup_size = 32
//      SM80: func.func @matmul_config_sm86()
// SM80-SAME:     translation_info = #[[TRANSLATION]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[32, 128, 32]]>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> ()>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @contract_reduction() {
  %c0 = arith.constant 0 : index
  %c40064 = arith.constant 40064 : index
  %c34752 = arith.constant 34752 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x7xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c40064) : !flow.dispatch.tensor<readonly:tensor<3x64x4x8xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c34752) : !flow.dispatch.tensor<writeonly:tensor<3x64xf32>>
  %3 = tensor.empty() : tensor<3x64xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 4], sizes = [3, 64, 4, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x64x4x8xf32>> -> tensor<3x64x4xf32>
  %5 = linalg.fill {lowering_config = #config} ins(%cst : f32) outs(%3 : tensor<3x64xf32>) -> tensor<3x64xf32>
  %6 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<3x7xf32>> -> tensor<f32>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%4, %6 : tensor<3x64x4xf32>, tensor<f32>) outs(%5 : tensor<3x64xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %8 = arith.subf %in, %in_0 : f32
    %9 = arith.maximumf %8, %cst : f32
    %10 = arith.mulf %9, %9 : f32
    %11 = arith.addf %out, %10 : f32
    linalg.yield %11 : f32
  } -> tensor<3x64xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [3, 64], strides = [1, 1] : tensor<3x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<3x64xf32>>
  return
}

//  SM80-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32
//      SM80: func.func @contract_reduction()
// SM80-SAME:     translation_info = #[[TRANSLATION]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @dynamic_pack_2x2() {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
  %4 = arith.index_castui %0 : i32 to index
  %5 = arith.index_castui %1 : i32 to index
  %6 = arith.index_castui %2 : i32 to index
  %7 = arith.index_castui %3 : i32 to index
  %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c64) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%4, %5}
  %9 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x?x2x2xi32>>{%6, %7}
  %10 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [%4, %5], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%4, %5} -> tensor<?x?xi32>
  %11 = tensor.empty(%6, %7) : tensor<?x?x2x2xi32>
  %pack = linalg.pack %10 inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %11 : tensor<?x?xi32> -> tensor<?x?x2x2xi32>
  flow.dispatch.tensor.store %pack, %9, offsets = [0, 0, 0, 0], sizes = [%6, %7, 2, 2], strides = [1, 1, 1, 1] : tensor<?x?x2x2xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x2x2xi32>>{%6, %7}
  return
}

//  SM80-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[16, 16]]>
//  SM80-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUPackUnPack workgroup_size = [32, 1, 1]>
//      SM80:   func.func @dynamic_pack_2x2()
// SM80-SAME:     translation_info = #[[TRANSLATION]]
//      SM80:     linalg.pack
// SM80-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @large_matmul_f16() {
  %cst = arith.constant 0.000000e+00 : f16
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<2560x1792xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<1792x2048xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<2560x2048xf16>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2560, 1792], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2560x1792xf16>> -> tensor<2560x1792xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1792, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1792x2048xf16>> -> tensor<1792x2048xf16>
  %5 = tensor.empty() : tensor<2560x2048xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2560x2048xf16>) -> tensor<2560x2048xf16>
  %7 = linalg.matmul ins(%3, %4 : tensor<2560x1792xf16>, tensor<1792x2048xf16>) outs(%6 : tensor<2560x2048xf16>) -> tensor<2560x2048xf16>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2560, 2048], strides = [1, 1] : tensor<2560x2048xf16> -> !flow.dispatch.tensor<writeonly:tensor<2560x2048xf16>>
  return
}
//  SM80-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128, 256, 32]{{\]}}
//  SM80-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCoreMmaSync workgroup_size = [128, 2, 1] subgroup_size = 32, {pipeline_depth = 3 : i64, store_stage = 1 : i64}>
//      SM80: func.func @large_matmul_f16()
// SM80-SAME:     translation_info = #[[TRANSLATION]]
//      SM80: linalg.fill
//      SM80: linalg.matmul
// SM80-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @large_matmul_f32() {
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<2560x1792xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<1792x2048xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<2560x2048xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2560, 1792], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2560x1792xf32>> -> tensor<2560x1792xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1792, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1792x2048xf32>> -> tensor<1792x2048xf32>
  %5 = tensor.empty() : tensor<2560x2048xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2560x2048xf32>) -> tensor<2560x2048xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<2560x1792xf32>, tensor<1792x2048xf32>) outs(%6 : tensor<2560x2048xf32>) -> tensor<2560x2048xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2560, 2048], strides = [1, 1] : tensor<2560x2048xf32> -> !flow.dispatch.tensor<writeonly:tensor<2560x2048xf32>>
  return
}

//  SM80-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128, 256, 16]{{\]}}
//  SM80-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCoreMmaSync workgroup_size = [128, 2, 1] subgroup_size = 32, {pipeline_depth = 4 : i64, store_stage = 1 : i64}>
//      SM80: func.func @large_matmul_f32()
// SM80-SAME:     translation_info = #[[TRANSLATION]]
//      SM80: linalg.fill
//      SM80: linalg.matmul
// SM80-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @inner_unit_dim() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<16384x1xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<16384x1xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<16384x1xf32>>
  %3 = tensor.empty() : tensor<16384x1xf32>
  %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16384, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16384x1xf32>> -> tensor<16384x1xf32>
  %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16384, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16384x1xf32>> -> tensor<16384x1xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %5 : tensor<16384x1xf32>, tensor<16384x1xf32>) outs(%3 : tensor<16384x1xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %7 = arith.addf %in, %in_0 : f32
    linalg.yield %7 : f32
  } -> tensor<16384x1xf32>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [16384, 1], strides = [1, 1] : tensor<16384x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<16384x1xf32>>
  return
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>
//      CHECK: func.func @inner_unit_dim()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #iree_gpu.lowering_config<{thread = [1, 1], workgroup = [32, 1]}>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3)>
func.func @forward_dispatch_1_conv_2d_nhwc_hwcf_256x112x112x64x7x7x3_f32() {
  %c0 = arith.constant 0 : index
  %c162508800 = arith.constant 162508800 : index
  %cst = arith.constant 1.001000e-05 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant dense_resource<__elided__> : tensor<64xf32>
  %cst_2 = arith.constant dense_resource<__elided__> : tensor<64xf32>
  %cst_3 = arith.constant dense_resource<__elided__> : tensor<64xf32>
  %cst_4 = arith.constant dense_resource<__elided__> : tensor<64xf32>
  %cst_5 = arith.constant dense_resource<__elided__> : tensor<64xf32>
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x230x230x3xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<7x7x3x64xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c162508800) : !flow.dispatch.tensor<writeonly:tensor<256x112x112x64xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [256, 230, 230, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<256x230x230x3xf32>> -> tensor<256x230x230x3xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [7, 7, 3, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<7x7x3x64xf32>> -> tensor<7x7x3x64xf32>
  %5 = tensor.empty() : tensor<256x112x112x64xf32>
  %6 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<256x230x230x3xf32>, tensor<7x7x3x64xf32>) outs(%6 : tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xf32>
  %8 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7, %cst_1, %cst_2, %cst_3, %cst_4, %cst_5 : tensor<256x112x112x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%5 : tensor<256x112x112x64xf32>) {
  ^bb0(%in: f32, %in_6: f32, %in_7: f32, %in_8: f32, %in_9: f32, %in_10: f32, %out: f32):
    %9 = arith.addf %in_9, %cst : f32
    %10 = math.sqrt %9 : f32
    %11 = arith.addf %in, %in_6 : f32
    %12 = arith.subf %11, %in_7 : f32
    %13 = arith.mulf %12, %in_8 : f32
    %14 = arith.divf %13, %10 : f32
    %15 = arith.addf %14, %in_10 : f32
    %16 = arith.maximumf %15, %cst_0 : f32
    linalg.yield %16 : f32
  } -> tensor<256x112x112x64xf32>
  flow.dispatch.tensor.store %8, %2, offsets = [0, 0, 0, 0], sizes = [256, 112, 112, 64], strides = [1, 1, 1, 1] : tensor<256x112x112x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<256x112x112x64xf32>>
  return
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1]
//      CHECK: func.func @forward_dispatch_1_conv_2d_nhwc_hwcf_256x112x112x64x7x7x3_f32
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.conv_2d
// CHECK-SAME:       lowering_config =  #iree_gpu.lowering_config<{promote_operands = [0, 1], reduction = [0, 0, 0, 0, 1, 7, 3], thread = [1, 1, 1, 1, 0, 0, 0], workgroup = [1, 1, 1, 32, 0, 0, 0]}>

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @_main_dispatch_15_generic_512x4x42x42x64_f32() {
  %cst = arith.constant 1.250000e-01 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %3 = arith.index_castui %0 {stream.alignment = 64 : index, stream.values = [35524672 : index, 240930880 : index, 446337088 : index, 651743296 : index]} : i32 to index
  %4 = arith.index_castui %1 {stream.alignment = 64 : index, stream.values = [57544768 : index, 262950976 : index, 468357184 : index, 673763392 : index]} : i32 to index
  %5 = arith.index_castui %2 {stream.alignment = 64 : index, stream.values = [1728 : index, 36472832 : index, 72943744 : index, 109415936 : index]} : i32 to index
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%3) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512x42x4x64xf32>>
  %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%4) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512x42x4x64xf32>>
  %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%5) : !flow.dispatch.tensor<writeonly:tensor<512x4x42x42xf32>>
  %9 = flow.dispatch.tensor.load %6, offsets = [0, 0, 0, 0], sizes = [512, 42, 4, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<512x42x4x64xf32>> -> tensor<512x42x4x64xf32>
  %10 = flow.dispatch.tensor.load %7, offsets = [0, 0, 0, 0], sizes = [512, 42, 4, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<512x42x4x64xf32>> -> tensor<512x42x4x64xf32>
  %11 = tensor.empty() : tensor<512x4x42x42xf32>
  %12 = linalg.fill ins(%cst_0 : f32) outs(%11 : tensor<512x4x42x42xf32>) -> tensor<512x4x42x42xf32>
  %13 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%9, %10 : tensor<512x42x4x64xf32>, tensor<512x42x4x64xf32>) outs(%12 : tensor<512x4x42x42xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %15 = arith.mulf %in, %in_1 : f32
    %16 = arith.addf %out, %15 : f32
    linalg.yield %16 : f32
  } -> tensor<512x4x42x42xf32>
  %14 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13 : tensor<512x4x42x42xf32>) outs(%11 : tensor<512x4x42x42xf32>) {
  ^bb0(%in: f32, %out: f32):
    %15 = arith.mulf %in, %cst : f32
    linalg.yield %15 : f32
  } -> tensor<512x4x42x42xf32>
  flow.dispatch.tensor.store %14, %8, offsets = [0, 0, 0, 0], sizes = [512, 4, 42, 42], strides = [1, 1, 1, 1] : tensor<512x4x42x42xf32> -> !flow.dispatch.tensor<writeonly:tensor<512x4x42x42xf32>>
  return
}

//       CHECK:  #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 8, 1] subgroup_size = 32, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>
//       CHECK:  func.func @_main_dispatch_15_generic_512x4x42x42x64_f32()
//  CHECK-SAME:    translation_info = #[[TRANSLATION]]
//       CHECK:  linalg.fill
//       CHECK:  linalg.generic
//  CHECK-SAME:     lowering_config = #iree_gpu.lowering_config<{reduction = [0, 0, 0, 0, 32], thread = [1, 1, 1, 16, 0], workgroup = [1, 1, 32, 128, 1]}>

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 9, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @i4_dequant_matvec() {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
  %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : i32
  %5 = hal.interface.constant.load layout(#pipeline_layout) ordinal(5) : i32
  %6 = hal.interface.constant.load layout(#pipeline_layout) ordinal(6) : i32
  %7 = hal.interface.constant.load layout(#pipeline_layout) ordinal(7) : i32
  %8 = hal.interface.constant.load layout(#pipeline_layout) ordinal(8) : i32
  %9 = arith.index_castui %0 : i32 to index
  %10 = arith.index_castui %1 : i32 to index
  %11 = arith.index_castui %2 : i32 to index
  %12 = arith.extui %3 : i32 to i64
  %13 = arith.extui %4 : i32 to i64
  %14 = arith.shli %13, %c32_i64 : i64
  %15 = arith.ori %12, %14 : i64
  %16 = arith.index_castui %15 : i64 to index
  %17 = arith.extui %5 : i32 to i64
  %18 = arith.extui %6 : i32 to i64
  %19 = arith.shli %18, %c32_i64 : i64
  %20 = arith.ori %17, %19 : i64
  %21 = arith.index_castui %20 : i64 to index
  %22 = arith.extui %7 : i32 to i64
  %23 = arith.extui %8 : i32 to i64
  %24 = arith.shli %23, %c32_i64 : i64
  %25 = arith.ori %22, %24 : i64
  %26 = arith.index_castui %25 : i64 to index
  %27 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%9) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x11008xi4>>
  %28 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%10) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096xf32>>
  %29 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%11) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096xf32>>
  %30 = flow.dispatch.workload.ordinal %26, 0 : index
  %31 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%16) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x11008xf32>>{%30}
  %32 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%21) : !flow.dispatch.tensor<writeonly:tensor<?x4096xf32>>{%30}
  %33 = flow.dispatch.tensor.load %27, offsets = [0, 0], sizes = [4096, 11008], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x11008xi4>> -> tensor<4096x11008xi4>
  %34 = flow.dispatch.tensor.load %28, offsets = [0], sizes = [4096], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4096xf32>> -> tensor<4096xf32>
  %35 = flow.dispatch.tensor.load %29, offsets = [0], sizes = [4096], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4096xf32>> -> tensor<4096xf32>
  %36 = flow.dispatch.tensor.load %31, offsets = [0, 0], sizes = [%30, 11008], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x11008xf32>>{%30} -> tensor<?x11008xf32>
  %37 = tensor.empty(%30) : tensor<?x4096xf32>
  %38 = tensor.empty() : tensor<4096x11008xf32>
  %39 = linalg.fill ins(%cst : f32) outs(%37 : tensor<?x4096xf32>) -> tensor<?x4096xf32>
  %40 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%33, %34, %35 : tensor<4096x11008xi4>, tensor<4096xf32>, tensor<4096xf32>) outs(%38 : tensor<4096x11008xf32>) {
  ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
    %42 = arith.extui %in : i4 to i32
    %43 = arith.uitofp %42 : i32 to f32
    %44 = arith.subf %43, %in_1 : f32
    %45 = arith.mulf %44, %in_0 : f32
    linalg.yield %45 : f32
  } -> tensor<4096x11008xf32>
  %41 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%36, %40 : tensor<?x11008xf32>, tensor<4096x11008xf32>) outs(%39 : tensor<?x4096xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %42 = arith.mulf %in, %in_0 : f32
    %43 = arith.addf %42, %out : f32
    linalg.yield %43 : f32
  } -> tensor<?x4096xf32>
  flow.dispatch.tensor.store %41, %32, offsets = [0, 0], sizes = [%30, 4096], strides = [1, 1] : tensor<?x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x4096xf32>>{%30}
  return
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 32]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [32, 1, 1]>
// CHECK-LABEL: func.func @i4_dequant_matvec()
//  CHECK-SAME:   translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     lowering_config = #[[$CONFIG]]
