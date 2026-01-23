// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | FileCheck %s

// CHECK:   LLVMGPUTileAndFuse
#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @elementwise_broadcast(%3: tensor<128256x4096xf16>, %4: tensor<1024xi64>) -> tensor<1024x4096xf16> {
  %5 = tensor.empty() : tensor<1024x4096xf16>
  %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<1024xi64>) outs(%5 : tensor<1024x4096xf16>) {
  ^bb0(%in: i64, %out: f16):
    %7 = linalg.index 1 : index
    %8 = arith.index_cast %in : i64 to index
    %extracted = tensor.extract %3[%8, %7] : tensor<128256x4096xf16>
    linalg.yield %extracted : f16
  } -> tensor<1024x4096xf16>
  return %6 : tensor<1024x4096xf16>
}
