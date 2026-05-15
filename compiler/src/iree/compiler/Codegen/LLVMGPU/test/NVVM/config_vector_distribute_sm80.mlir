// RUN: iree-opt --split-input-file --iree-gpu-test-target=sm_80 \
// RUN:   --iree-codegen-llvmgpu-use-vector-distribution --iree-codegen-llvmgpu-use-tile-and-fuse-matmul=false \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

// Test that sm_80 selects NV_MMA_SYNC intrinsics for matmul operations.
// With --iree-codegen-llvmgpu-use-vector-distribution, it should select #iree_gpu.pipeline<VectorDistribute>.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_256x256x256_f16_f32() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %5 = tensor.empty() : tensor<256x256xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  return
}

// CHECK:      #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
// Verify that the matmul gets NV_MMA_SYNC_F32 intrinsic.
// CHECK-LABEL: func.func @matmul_256x256x256_f16_f32()
// CHECK:       linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
// CHECK-SAME:                mma_kind = #iree_gpu.mma_layout<NV_MMA_SYNC_F32_16x8x16_F16>

// -----

// Test F16 output matmul also selects NV_MMA_SYNC_F16 intrinsic.

#pipeline_layout_f16 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_256x256x256_f16_f16() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf16>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %5 = tensor.empty() : tensor<256x256xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<256x256xf16>) -> tensor<256x256xf16>
  %7 = linalg.matmul ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf16>) -> tensor<256x256xf16>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf16>>
  return
}

// CHECK:      #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
// Verify that the f16 output matmul gets NV_MMA_SYNC_F16 intrinsic.
// CHECK-LABEL: func.func @matmul_256x256x256_f16_f16()
// CHECK:       linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
// CHECK-SAME:                mma_kind = #iree_gpu.mma_layout<NV_MMA_SYNC_F16_16x8x16_F16>

// -----

// f64 reduction on sm_80: only AMDGPU has a `gpu.shuffle` lowering that
// decomposes wider element types into 32-bit chunks, so the gate (positive
// `target.isAMD()` check) rejects 64-bit on sm_80 and the dispatch must NOT
// pick VectorDistribute. Once the NVVM lowering grows the same support
// (llvm/llvm-project#197080, cf. the AMD precedent at
// llvm/llvm-project#136320), the gate can broaden to include sm_*.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @reduction_f64_sm80_no_vector_distribute() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f64
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4096xf64>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf64>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4096xf64>> -> tensor<8x4096xf64>
  %3 = tensor.empty() : tensor<8xf64>
  %4 = linalg.fill ins(%cst : f64) outs(%3 : tensor<8xf64>) -> tensor<8xf64>
  %5 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]}
      ins(%2 : tensor<8x4096xf64>) outs(%4 : tensor<8xf64>) {
      ^bb0(%in: f64, %out: f64):
        %6 = arith.addf %in, %out : f64
        linalg.yield %6 : f64
      } -> tensor<8xf64>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf64> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf64>>
  return
}

// CHECK-LABEL: func.func @reduction_f64_sm80_no_vector_distribute
// CHECK-NOT:   pipeline = #iree_gpu.pipeline<VectorDistribute>

// -----

// i48 (non-power-of-two) reduction: the bitwidth gate's power-of-two check
// rejects this independent of the target, so VectorDistribute must NOT be
// chosen. Co-located with the f64 fallback test rather than the AMDGPU file
// because the property under test is target-independent.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @reduction_i48_falls_back() {
  %c0 = arith.constant 0 : index
  %czero = arith.constant 0 : i48
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4096xi48>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xi48>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4096xi48>> -> tensor<8x4096xi48>
  %3 = tensor.empty() : tensor<8xi48>
  %4 = linalg.fill ins(%czero : i48) outs(%3 : tensor<8xi48>) -> tensor<8xi48>
  %5 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]}
      ins(%2 : tensor<8x4096xi48>) outs(%4 : tensor<8xi48>) {
      ^bb0(%in: i48, %out: i48):
        %6 = arith.addi %in, %out : i48
        linalg.yield %6 : i48
      } -> tensor<8xi48>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xi48> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xi48>>
  return
}

// CHECK-LABEL: func.func @reduction_i48_falls_back
// CHECK-NOT:   pipeline = #iree_gpu.pipeline<VectorDistribute>
