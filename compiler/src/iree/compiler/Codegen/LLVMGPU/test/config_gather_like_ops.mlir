// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | FileCheck %s

// CHECK:   LLVMGPUTileAndFuse
#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module {
  func.func @elementwise_broadcast() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128256x4096xf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xi64>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x4096xf16>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128256, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128256x4096xf16>> -> tensor<128256x4096xf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [1024], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xi64>> -> tensor<1024xi64>
    %5 = tensor.empty() : tensor<1024x4096xf16>
    %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<1024xi64>) outs(%5 : tensor<1024x4096xf16>) {
    ^bb0(%in: i64, %out: f16):
      %7 = linalg.index 1 : index
      %8 = arith.index_cast %in : i64 to index
      %extracted = tensor.extract %3[%8, %7] : tensor<128256x4096xf16>
      linalg.yield %extracted : f16
    } -> tensor<1024x4096xf16>
    iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [1024, 4096], strides = [1, 1] : tensor<1024x4096xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x4096xf16>>
    return
  }
}
