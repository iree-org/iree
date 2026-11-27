// RUN: iree-compile --iree-hal-target-backends=vulkan-spirv --iree-vulkan-target=vp_android_baseline_2022 --compile-to=executable-targets %s | FileCheck %s

// Attention must decompose to supported operations before bufferisation.
// The full pipeline must produce a target executable for a masked partial tile.
// CHECK: hal.executable.variant
// CHECK-NOT: iree_linalg_ext.{{(online_)?attention}}
func.func @masked_attention(%query: tensor<1x3x4xf32>, %key: tensor<1x5x4xf32>, %value: tensor<1x5x4xf32>, %mask: tensor<1x3x5xf32>) -> tensor<1x3x4xf32> {
  %init = tensor.empty() : tensor<1x3x4xf32>
  %scale = arith.constant 0.5 : f32
  %result = iree_linalg_ext.attention <indexing_maps = [
      affine_map<(b, m, k, n, v) -> (b, m, k)>,
      affine_map<(b, m, k, n, v) -> (b, n, k)>,
      affine_map<(b, m, k, n, v) -> (b, n, v)>,
      affine_map<(b, m, k, n, v) -> ()>,
      affine_map<(b, m, k, n, v) -> (b, m, n)>,
      affine_map<(b, m, k, n, v) -> (b, m, v)>]>
      ins(%query, %key, %value, %scale, %mask : tensor<1x3x4xf32>, tensor<1x5x4xf32>, tensor<1x5x4xf32>, f32, tensor<1x3x5xf32>)
      outs(%init : tensor<1x3x4xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<1x3x4xf32>
  return %result : tensor<1x3x4xf32>
}
