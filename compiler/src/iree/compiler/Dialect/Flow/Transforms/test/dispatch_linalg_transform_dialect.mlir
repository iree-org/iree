// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-flow-dispatch-with-transform-dialect{transform-file-name=%p/transform_dialect_dispatch_spec.mlir}))" %s | \
// RUN: FileCheck %s

func.func @tile_matmul_alone(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
                             %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//      CHECK: func.func @tile_matmul_alone
//      CHECK:   flow.dispatch.workgroups

func.func @tile_matmul_with_constant(
    %arg1 : tensor<5x10xf32>, %arg2 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  // The constant is cloned and fused into the dispatch region.
  %a = arith.constant dense<1.0> : tensor<10x5xf32>
  %1 = linalg.matmul ins(%a, %arg1 : tensor<10x5xf32>, tensor<5x10xf32>)
    outs(%arg2 : tensor<10x10xf32>) -> tensor<10x10xf32>
  return %1 : tensor<10x10xf32>
}
//      CHECK: func.func @tile_matmul_with_constant
//      CHECK:   flow.dispatch.workgroups
//      CHECK:     arith.constant dense<1.000000e+00> : tensor<10x5xf32>

// Some dummy functions to exercise TSAN under parallelism.
func.func @foo1() -> index {
  %0 = arith.constant 1 : index
  return %0 : index
}
func.func @foo2() -> index {
  %0 = arith.constant 2 : index
  return %0 : index
}
func.func @foo3() -> index {
  %0 = arith.constant 3 : index
  return %0 : index
}
func.func @foo4() -> index {
  %0 = arith.constant 4 : index
  return %0 : index
}
func.func @foo5() -> index {
  %0 = arith.constant 5 : index
  return %0 : index
}
