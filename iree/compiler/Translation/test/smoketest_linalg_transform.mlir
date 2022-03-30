// RUN: iree-compile %s -iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=dylib-llvm-aot \
// RUN:   -iree-codegen-use-linalg-transform-interp -linalg-transform-file-name=%p/linalg_transform_spec.mlir | \
// RUN: iree-check-module  --driver=dylib -

func @matmul_static() {
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

  // util.do_not_optimize on output to prevent fusing in the same dispatch
  // region which would be subject to racy tensor semantics.
  // Forcing different dispatches forces flow.dispatch.tensor.load which is
  // actually side-effecting.
  %res_in = util.do_not_optimize(%res) : tensor<5x5xf32>
  %matmul = linalg.matmul
      ins(%lhs, %rhs : tensor<5x3xf32>, tensor<3x5xf32>)
      outs(%res_in : tensor<5x5xf32>) -> tensor<5x5xf32>
  %matmul_res = util.do_not_optimize(%matmul) : tensor<5x5xf32>

  check.expect_almost_eq_const(%matmul_res,
    dense<[[430.0, 388.0, 346.0, 304.0, 262.0],
           [340.0, 307.0, 274.0, 241.0, 208.0],
           [250.0, 226.0, 202.0, 178.0, 154.0],
           [160.0, 145.0, 130.0, 115.0, 100.0],
           [70.0, 64.0, 58.0, 52.0, 46.0]]> : tensor<5x5xf32>) : tensor<5x5xf32>
  return
}
