// RUN: iree-opt --split-input-file --iree-util-test-trivial-constexpr-folding %s | FileCheck %s

// CHECK-LABEL func @linalgConstantTranspose
func.func @linalgConstantTranspose() -> (tensor<20480x5120xf16>) {
  %cst = arith.constant dense_resource<__elided__> : tensor<5120x20480xf16>
  %empty= tensor.empty() : tensor<20480x5120xf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} 
    ins(%cst : tensor<5120x20480xf16>) outs(%empty : tensor<20480x5120xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<20480x5120xf16>
  return %1 : tensor<20480x5120xf16>
}
