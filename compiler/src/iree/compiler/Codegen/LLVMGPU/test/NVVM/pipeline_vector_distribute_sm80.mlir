// RUN: iree-opt --split-input-file --iree-gpu-test-target=sm_80 \
// RUN:   --iree-codegen-llvmgpu-nvvm-lowering-pipeline='include-llvm-lowering=false' \
// RUN:   %s | FileCheck %s

// Test matmul lowering with NV_MMA_SYNC intrinsics produces nvgpu.mma.sync operations.

#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#config = #iree_gpu.lowering_config<{workgroup = [32, 16, 0], reduction = [0, 0, 32], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<NV_MMA_SYNC_F32_16x8x16_F16>, subgroup_basis = [[2, 2, 1], [0, 1, 2]]}>
#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 2, 1] subgroup_size = 32, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 1, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_256x256x256_f16_f32() attributes {hal.executable.target = #executable_target_cuda, translation_info = #translation} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %5 = tensor.empty() : tensor<256x256xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
  %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  return
}

// Basic pipeline test to make sure it generates nvgpu.mma.sync instructions.
// With workgroup=[32,16], reduction=[0,0,32], each workgroup handles a 32x16 output tile.
// With 2x2 subgroups (m_count=2, n_count=2), each subgroup handles 16x8.
// K=256 with reduction tile K=32 means 8 loop iterations, 2 mma.sync per iteration.

// CHECK-LABEL: func.func @matmul_256x256x256_f16_f32()
//       CHECK:   scf.for {{.*}} = %c0 to %c256 step %c32 iter_args({{.*}}) -> (vector<2x2xf32>)
// CHECK-COUNT-2:   nvgpu.mma.sync({{.*}}) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
//       CHECK:     scf.yield

// -----

// Test with F16 accumulator

#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#config_f16 = #iree_gpu.lowering_config<{workgroup = [32, 16, 0], reduction = [0, 0, 32], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<NV_MMA_SYNC_F16_16x8x16_F16>, subgroup_basis = [[2, 2, 1], [0, 1, 2]]}>
#translation_f16 = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 2, 1] subgroup_size = 32, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 1, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout_f16 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_256x256x256_f16_f16() attributes {hal.executable.target = #executable_target_cuda, translation_info = #translation_f16} {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf16>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %5 = tensor.empty() : tensor<256x256xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %7 = linalg.matmul {lowering_config = #config_f16} ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf16>) -> tensor<256x256xf16>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf16>>
  return
}

// Test F16 accumulator path - should use NV_MMA_SYNC_F16_16x8x16_F16 and produce f16 results.

// CHECK-LABEL: func.func @matmul_256x256x256_f16_f16()
//       CHECK:   scf.for {{.*}} = %c0 to %c256 step %c32 iter_args({{.*}}) -> (vector<2x2xf16>)
// CHECK-COUNT-2:   nvgpu.mma.sync({{.*}}) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
//       CHECK:     scf.yield
