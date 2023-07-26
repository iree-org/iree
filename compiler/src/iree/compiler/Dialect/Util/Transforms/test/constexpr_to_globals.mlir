// RUN: iree-opt --split-input-file --iree-util-const-expr-to-globals %s | FileCheck %s

// CHECK-LABEL: module @linalgConstantTranspose
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
module @linalgConstantTranspose {
  func.func @main() -> tensor<20480x5120xf16> {
    %0 = util.constexpr  : () -> tensor<20480x5120xf16> {
      %cst = arith.constant dense_resource<__elided__> : tensor<5120x20480xf16>
      %1 = tensor.empty() : tensor<20480x5120xf16>
      %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<5120x20480xf16>) outs(%1 : tensor<20480x5120xf16>) {
      ^bb0(%in: f16, %out: f16):
        linalg.yield %in : f16
      } -> tensor<20480x5120xf16>
      util.yield_const %2 : tensor<20480x5120xf16>
    }
    return %0 : tensor<20480x5120xf16>
  }
}

// -----

module @initializer {
  util.global private @constexpr_143 : tensor<5120x50688xf16>
  util.initializer {
    %0 = util.constexpr  : () -> tensor<5120x50688xf16> {
      %cst = arith.constant dense_resource<__elided__> : tensor<50688x5120xf16>
      %1 = tensor.empty() : tensor<5120x50688xf16>
      %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<50688x5120xf16>) outs(%1 : tensor<5120x50688xf16>) {
      ^bb0(%in: f16, %out: f16):
        linalg.yield %in : f16
      } -> tensor<5120x50688xf16>
      util.yield_const %2 : tensor<5120x50688xf16>
    }
    util.global.store %0, @constexpr_143 : tensor<5120x50688xf16>
    util.initializer.return
  }
}
