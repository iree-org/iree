// R-UN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu \
/// Specify the dispatch region formation with the transform dialect.
// R-UN:   --iree-flow-dispatch-use-transform-dialect=%p/transform_dialect_dispatch_spec.mlir \
/// Specify the codegen strategy with the transform dialect.
// R-UN:   --iree-codegen-llvmcpu-use-transform-dialect=%p/transform_dialect_codegen_spec.mlir \
// R-UN: %s | FileCheck %s


// RUN: iree-opt %s \
// RUN:   --iree-abi-transformation-pipeline \
// RUN:   --iree-flow-transformation-pipeline \
// RUN:   --iree-flow-dispatch-use-transform-dialect=%p/transform_dialect_dispatch_spec.mlir

func.func @matmul_static() -> tensor<5x5xf32> {
  %res = flow.tensor.constant dense<[
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0]]> : tensor<5x5xf32> -> tensor<5x5xf32>
  %lhs = flow.tensor.constant dense<[
    [15.0, 14.0, 13.0],
    [12.0, 11.0, 10.0],
    [09.0, 08.0, 07.0],
    [06.0, 05.0, 04.0],
    [03.0, 02.0, 01.0]]> : tensor<5x3xf32> -> tensor<5x3xf32>
  %rhs = flow.tensor.constant dense<[
    [15.0, 14.0, 13.0, 12.0, 11.0],
    [10.0, 09.0, 08.0, 07.0, 06.0],
    [05.0, 04.0, 03.0, 02.0, 01.0]]> : tensor<3x5xf32> -> tensor<3x5xf32>

  %matmul = linalg.matmul
      ins(%lhs, %rhs : tensor<5x3xf32>, tensor<3x5xf32>)
      outs(%res : tensor<5x5xf32>) -> tensor<5x5xf32>
  %matmul_res = util.optimization_barrier %matmul : tensor<5x5xf32>

  return %matmul_res : tensor<5x5xf32>
}

//      CHECK: 5x5xf32=
// CHECK-SAME: [430 388 346 304 262]
// CHECK-SAME: [340 307 274 241 208]
// CHECK-SAME: [250 226 202 178 154]
// CHECK-SAME: [160 145 130 115 100]
// CHECK-SAME: [70 64 58 52 46]
