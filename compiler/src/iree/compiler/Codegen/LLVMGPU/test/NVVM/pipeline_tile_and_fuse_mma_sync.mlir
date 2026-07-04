// RUN: iree-opt --split-input-file \
// RUN:   --iree-codegen-llvmgpu-nvvm-lowering-pipeline='include-llvm-lowering=false' %s | FileCheck %s

// Test that TileAndFuse pipeline with NV_MMA_SYNC generates nvgpu.mma.sync operations.

#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{
  workgroup = [64, 64, 0],
  reduction = [0, 0, 8],
  subgroup = [2, 4],
  mma_kind = #iree_gpu.mma_layout<NV_MMA_SYNC_F32_16x8x16_F16>,
  promote_operands = [0, 1]
}>
func.func @matmul_tile_and_fuse_mma_sync()
  attributes {
    hal.executable.target = #executable_target_cuda,
    translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [128, 1, 1] subgroup_size = 32>} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1280xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x10240xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1280, 10240], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x10240xf16>> -> tensor<1280x10240xf16>
  %5 = tensor.empty() : tensor<2048x10240xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
  %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<2048x1280xf16>, tensor<1280x10240xf16>) outs(%6 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
  return
}

// CHECK-LABEL: func @matmul_tile_and_fuse_mma_sync
//   CHECK-DAG:   %[[B0:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[B1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[B2:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//   CHECK-DAG:   memref.alloc() : memref<{{.*}}xf16, #gpu.address_space<workgroup>>
//   CHECK-DAG:   memref.alloc() : memref<{{.*}}xf16, #gpu.address_space<workgroup>>
//       CHECK:   scf.for
//       CHECK:       gpu.barrier
//       CHECK:       vector.transfer_read {{.*}} vector<{{.*}}xf16>
//       CHECK:       vector.transfer_write
//       CHECK:       gpu.barrier
// Verify LHS transpose for mma.sync column-major ordering
// TileAndFuse produces 2x1x2x2 -> reshape to 2x2x2 -> transpose [1,0,2] -> reshape to 4x2
//       CHECK:       vector.shape_cast {{.*}} : vector<2x1x2x2xf16> to vector<2x2x2xf16>
//       CHECK:       vector.transpose {{.*}}, [1, 0, 2] : vector<2x2x2xf16> to vector<2x2x2xf16>
//       CHECK:       vector.shape_cast {{.*}} : vector<2x2x2xf16> to vector<4x2xf16>
// Verify nvgpu.mma.sync is generated with correct shape
// CHECK-COUNT-8: nvgpu.mma.sync({{.*}}) {mmaShape = [16, 8, 16]}
//       CHECK:   scf.yield

// -----

// Test with F16 accumulator

#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout_f16 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config_f16 = #iree_gpu.lowering_config<{
  workgroup = [64, 64, 0],
  reduction = [0, 0, 8],
  subgroup = [2, 4],
  mma_kind = #iree_gpu.mma_layout<NV_MMA_SYNC_F16_16x8x16_F16>,
  promote_operands = [0, 1]
}>
func.func @matmul_tile_and_fuse_mma_sync_f16()
  attributes {
    hal.executable.target = #executable_target_cuda,
    translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [128, 1, 1] subgroup_size = 32>} {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1280xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x10240xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x10240xf16>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1280, 10240], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x10240xf16>> -> tensor<1280x10240xf16>
  %5 = tensor.empty() : tensor<2048x10240xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2048x10240xf16>) -> tensor<2048x10240xf16>
  %7 = linalg.matmul {lowering_config = #config_f16} ins(%3, %4 : tensor<2048x1280xf16>, tensor<1280x10240xf16>) outs(%6 : tensor<2048x10240xf16>) -> tensor<2048x10240xf16>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x10240xf16>>
  return
}

// CHECK-LABEL: func @matmul_tile_and_fuse_mma_sync_f16
// Verify LHS transpose for mma.sync column-major ordering (same pattern as f32)
//       CHECK:       vector.shape_cast {{.*}} : vector<2x1x2x2xf16> to vector<2x2x2xf16>
//       CHECK:       vector.transpose {{.*}}, [1, 0, 2] : vector<2x2x2xf16> to vector<2x2x2xf16>
//       CHECK:       vector.shape_cast {{.*}} : vector<2x2x2xf16> to vector<4x2xf16>
// Verify nvgpu.mma.sync is generated with f16 output type
// CHECK-COUNT-8: nvgpu.mma.sync({{.*}}) {mmaShape = [16, 8, 16]} : ({{.*}}) -> vector<2x2xf16>

// -----

// Test BF16 matmul lowering through TileAndFuse pipeline with NV_MMA_SYNC_F32_16x8x16_BF16.

#executable_target_cuda_bf16 = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout_bf16 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config_bf16 = #iree_gpu.lowering_config<{
  workgroup = [64, 64, 0],
  reduction = [0, 0, 8],
  subgroup = [2, 4],
  mma_kind = #iree_gpu.mma_layout<NV_MMA_SYNC_F32_16x8x16_BF16>,
  promote_operands = [0, 1]
}>
func.func @matmul_tile_and_fuse_mma_sync_bf16()
  attributes {
    hal.executable.target = #executable_target_cuda_bf16,
    translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [128, 1, 1] subgroup_size = 32>} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_bf16) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1280xbf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_bf16) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x10240xbf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_bf16) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1280xbf16>> -> tensor<2048x1280xbf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1280, 10240], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x10240xbf16>> -> tensor<1280x10240xbf16>
  %5 = tensor.empty() : tensor<2048x10240xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
  %7 = linalg.matmul {lowering_config = #config_bf16} ins(%3, %4 : tensor<2048x1280xbf16>, tensor<1280x10240xbf16>) outs(%6 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
  return
}

// CHECK-LABEL: func @matmul_tile_and_fuse_mma_sync_bf16
//   CHECK-DAG:   memref.alloc() : memref<{{.*}}xbf16, #gpu.address_space<workgroup>>
//   CHECK-DAG:   memref.alloc() : memref<{{.*}}xbf16, #gpu.address_space<workgroup>>
//       CHECK:   scf.for
// Verify nvgpu.mma.sync is generated with correct shape for BF16
// CHECK-COUNT-8: nvgpu.mma.sync({{.*}}) {mmaShape = [16, 8, 16]} : ({{.*}}) -> vector<2x2xf32>
//       CHECK:   scf.yield

// -----

// Test FP8 (E4M3FN) matmul lowering through TileAndFuse pipeline.
// m16n8k32: K doubles vs F16/BF16, so reduction tile = 4 (4x32=128 K-per-iter).
// LHS transpose reshapes to {outer_M, outer_K, element_K} = {2, 2, 4} for FP8
// (vs {2, 2, 2} for FP16 m16n8k16).

#executable_target_cuda_fp8 = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout_fp8 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config_fp8e4m3fn = #iree_gpu.lowering_config<{
  workgroup = [64, 64, 0],
  reduction = [0, 0, 4],
  subgroup = [2, 4],
  mma_kind = #iree_gpu.mma_layout<NV_MMA_SYNC_F32_16x8x32_F8E4M3FN>,
  promote_operands = [0, 1]
}>
func.func @matmul_tile_and_fuse_mma_sync_fp8e4m3fn()
  attributes {
    hal.executable.target = #executable_target_cuda_fp8,
    translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [128, 1, 1] subgroup_size = 32>} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_fp8) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1280xf8E4M3FN>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_fp8) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x10240xf8E4M3FN>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_fp8) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1280xf8E4M3FN>> -> tensor<2048x1280xf8E4M3FN>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1280, 10240], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x10240xf8E4M3FN>> -> tensor<1280x10240xf8E4M3FN>
  %5 = tensor.empty() : tensor<2048x10240xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
  %7 = linalg.matmul {lowering_config = #config_fp8e4m3fn} ins(%3, %4 : tensor<2048x1280xf8E4M3FN>, tensor<1280x10240xf8E4M3FN>) outs(%6 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
  return
}

// CHECK-LABEL: func @matmul_tile_and_fuse_mma_sync_fp8e4m3fn
//   CHECK-DAG:   memref.alloc() : memref<{{.*}}xf8E4M3FN, #gpu.address_space<workgroup>>
//   CHECK-DAG:   memref.alloc() : memref<{{.*}}xf8E4M3FN, #gpu.address_space<workgroup>>
//       CHECK:   scf.for
// Verify LHS transpose for mma.sync column-major ordering (FP8 element[K]=4)
// TileAndFuse produces 2x1x2x4 -> reshape to 2x2x4 -> transpose [1,0,2] -> reshape to 4x4
//       CHECK:       vector.shape_cast {{.*}} : vector<2x1x2x4xf8E4M3FN> to vector<2x2x4xf8E4M3FN>
//       CHECK:       vector.transpose {{.*}}, [1, 0, 2] : vector<2x2x4xf8E4M3FN> to vector<2x2x4xf8E4M3FN>
//       CHECK:       vector.shape_cast {{.*}} : vector<2x2x4xf8E4M3FN> to vector<4x4xf8E4M3FN>
// Verify nvgpu.mma.sync is generated with m16n8k32 shape for FP8
// CHECK-COUNT-8: nvgpu.mma.sync({{.*}}) {mmaShape = [16, 8, 32]} : ({{.*}}) -> vector<2x2xf32>
//       CHECK:   scf.yield
