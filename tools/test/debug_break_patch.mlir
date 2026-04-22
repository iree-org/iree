// Patch module consumed by tools/test/debug_break.mlir. At each break the
// instrumentation re-parses this file and splices it into the live module.
// The shape (util.func / util.return / tensor args) must match what the IR
// looks like at the break point — after iree-sanitize-module-names and
// before iree-abi-wrap-entry-points. This patch renames @add -> @patched_add
// and replaces addf with mulf so that later passes (ABI wrapping etc.) run
// on the patched IR and produce visibly different output.
module {
  util.func public @patched_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
    util.return %0 : tensor<4xf32>
  }
}
