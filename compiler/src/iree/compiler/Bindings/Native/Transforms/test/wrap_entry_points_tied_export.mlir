// RUN: iree-opt --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline='builtin.module(iree-import-public,iree-abi-wrap-entry-points)' %s

// A result tie on a public func.func reaches native wrapping as a tied export.
// expected-error @+1 {{native ABI exports do not support tied results; use iree.abi.output on an argument to provide result storage (result 1)}}
func.func @tied_func(%arg0: tensor<4x3xi32>, %arg1: tensor<2x5xf32>) -> (tensor<2x5xf32>, tensor<4x3xi32> {iree.abi.tied = 0 : i64}) {
  return %arg1, %arg0 : tensor<2x5xf32>, tensor<4x3xi32>
}

// -----

// expected-error @+1 {{native ABI exports do not support tied results; use iree.abi.output on an argument to provide result storage (result 0)}}
util.func public @tied_util(%arg0: tensor<4x3xi32>, %arg1: tensor<2x5xf32>) -> %arg0 {
  util.return %arg0 : tensor<4x3xi32>
}
