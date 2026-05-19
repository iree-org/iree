// RUN: iree-opt --iree-gpu-test-target=gfx942 \
// RUN:   --iree-codegen-gpu-enable-vector-alloc-swizzle \
// RUN:   --iree-codegen-llvmgpu-rocdl-lowering-pipeline \
// RUN:   %s | FileCheck %s

#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 64], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>, subgroup_basis = [[2, 2, 1], [0, 1, 2]]}>
#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @matmul_256x256x256_f32() attributes {hal.executable.target = #executable_target_rocm, translation_info = #translation} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf32>> -> tensor<256x256xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf32>> -> tensor<256x256xf32>
  %5 = tensor.empty() : tensor<256x256xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
  %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<256x256xf32>, tensor<256x256xf32>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  return
}

// CHECK-DAG: llvm.mlir.global private @{{__shared_memory.*}}() {{.*}} : !llvm.array<4096 x f32>
// CHECK-DAG: llvm.mlir.global private @{{__shared_memory.*}}() {{.*}} : !llvm.array<4096 x f32>
// CHECK-DAG: llvm.mlir.global private @{{__shared_memory.*}}() {{.*}} : !llvm.array<128 x array<68 x f16>>
// CHECK-DAG: llvm.mlir.global private @{{__shared_memory.*}}() {{.*}} : !llvm.array<64 x array<132 x f16>>
// CHECK-LABEL: llvm.func @matmul_256x256x256_f32
// CHECK-NOT: iree_codegen.swizzle_hint
// CHECK-NOT: xor_shuffle
// CHECK: llvm.xor
// CHECK-COUNT-64: rocdl.mfma.f32.16x16x4f32
// CHECK-NOT: iree_codegen.swizzle_hint
// CHECK-NOT: xor_shuffle
// CHECK: llvm.return

#config_f16 = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 128], promote_operands = [0, 1], mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_basis = [[2, 2, 1], [0, 1, 2]]}>
#translation_f16 = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout_f16 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @matmul_256x256x256_f16_f32() attributes {hal.executable.target = #executable_target_rocm, translation_info = #translation_f16} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %5 = tensor.empty() : tensor<256x256xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
  %7 = linalg.matmul {lowering_config = #config_f16} ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  return
}

// Sub-bank f16 with mismatched writer/reader accesses should not use the
// swizzle-hint path. It should fall through to the existing bank-conflict
// padding path, visible above as padded LDS globals with inner sizes 68 and 132.
// CHECK-LABEL: llvm.func @matmul_256x256x256_f16_f32
// CHECK-NOT: iree_codegen.swizzle_hint
// CHECK-NOT: xor_shuffle
// CHECK-NOT: llvm.xor
// CHECK-COUNT-32: rocdl.mfma.f32.16x16x16f16
// CHECK-NOT: llvm.xor
// CHECK: llvm.return
