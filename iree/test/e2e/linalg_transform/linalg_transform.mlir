// RUN: iree-run-mlir %s --iree-hal-target-backends=dylib-llvm-aot -iree-codegen-use-linalg-transform-interp -linalg-transform-file-name=%p/linalg_transform_spec.mlir 

// TODO: Atm this cannot be correct: 
//   1. dispatch region formation creates a dispatch region that operates on the whole tensors.
//   2. the result tensor is produced by a constant op that is in the same dispatch region at the time of codegen.
//   3. the result buffer initial value is a memref.global constant.
//   4. bufferization properly introduces a copy of the `%res` constant buffer.
//   5. unfortunately the copy is in a mixed sequential/parallel context and becomes racy.
//   5. only one thread writes back the whole result.
// The solution is to properly create dispatch regions from the InParallel semantics 
// that don't mix sequential semantics  at the boundaries with distributed and 
// parallel semantics inside.
// NORUN: iree-run-mlir %s --iree-hal-target-backends=dylib-llvm-aot -iree-codegen-use-linalg-transform-interp -linalg-transform-file-name=%p/linalg_transform_inparallel_buffers_spec.mlir 

// TODO: Atm this one works only because of the spurious extra copies introduced in the tensor path.
// The solution is again to properly create dispatch regions from the InParallel semantics to avoid 
// implicit bufferization assumptions.
// RUN: iree-run-mlir %s --iree-hal-target-backends=dylib-llvm-aot -iree-codegen-use-linalg-transform-interp -linalg-transform-file-name=%p/linalg_transform_inparallel_buffers_spec.mlir 

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
  %matmul_res = util.do_not_optimize(%matmul) : tensor<5x5xf32>

  return %matmul_res : tensor<5x5xf32>
}

//      CHECK: 5x5xf32=
// CHECK-SAME: [430 388 346 304 262]
// CHECK-SAME: [340 307 274 241 208]
// CHECK-SAME: [250 226 202 178 154]
// CHECK-SAME: [160 145 130 115 100]
// CHECK-SAME: [70 64 58 52 46]
