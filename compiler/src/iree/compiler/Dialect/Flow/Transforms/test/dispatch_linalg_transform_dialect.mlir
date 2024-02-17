// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-flow-dispatch-with-transform-dialect{transform-file-name=%p/transform_dialect_dispatch_spec.mlir}))" %s | \
// RUN: FileCheck %s

util.func public @tile_matmul_alone(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
                             %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %1 : tensor<?x?xf32>
}
//      CHECK: util.func public @tile_matmul_alone
//      CHECK:   flow.dispatch.workgroups

util.func public @tile_matmul_with_constant(
    %arg1 : tensor<5x10xf32>, %arg2 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  // The constant is cloned and fused into the dispatch region.
  %a = arith.constant dense<1.0> : tensor<10x5xf32>
  %1 = linalg.matmul ins(%a, %arg1 : tensor<10x5xf32>, tensor<5x10xf32>)
    outs(%arg2 : tensor<10x10xf32>) -> tensor<10x10xf32>
  util.return %1 : tensor<10x10xf32>
}
//      CHECK: util.func public @tile_matmul_with_constant
//      CHECK:   flow.dispatch.workgroups
//      CHECK:     arith.constant dense<1.000000e+00> : tensor<10x5xf32>

// Some dummy functions to exercise TSAN under parallelism.
util.func public @foo1() -> index {
  %0 = arith.constant 1 : index
  util.return %0 : index
}
util.func public @foo2() -> index {
  %0 = arith.constant 2 : index
  util.return %0 : index
}
util.func public @foo3() -> index {
  %0 = arith.constant 3 : index
  util.return %0 : index
}
util.func public @foo4() -> index {
  %0 = arith.constant 4 : index
  util.return %0 : index
}
util.func public @foo5() -> index {
  %0 = arith.constant 5 : index
  util.return %0 : index
}
