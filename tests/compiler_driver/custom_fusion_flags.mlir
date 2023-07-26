// RUN: iree-compile --iree-hal-target-backends=llvm-cpu --compile-to=flow \
// RUN:   --iree-flow-custom-fusion-pass-pipeline="builtin.module(func.func(linalg-generalize-named-ops))" \
// RUN:   --mlir-print-ir-after=linalg-generalize-named-ops %s 2>&1 \
// RUN:   | FileCheck %s

func.func @test(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>, %arg2 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<10x10xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<10x10xf32>) -> tensor<10x10xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<10x10xf32>, tensor<10x10xf32>)
      outs(%fill : tensor<10x10xf32>) -> tensor<10x10xf32>
  %empty_2 = tensor.empty() : tensor<10x10xf32>
  %fill_2 = linalg.fill ins(%cst : f32) outs(%empty_2 : tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = linalg.matmul ins(%0, %arg2 : tensor<10x10xf32>, tensor<10x10xf32>)
      outs(%fill_2 : tensor<10x10xf32>) -> tensor<10x10xf32>
  return %1 : tensor<10x10xf32>
}
// Just check that the pass runs, and that the compilation finishes
//       CHECK: LinalgGeneralization (linalg-generalize-named-ops)
// CHECK-LABEL: module
// CHECK-NOT:   linalg.matmul
// CHECK:       linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]
// CHECK-NOT:   linalg.matmul
