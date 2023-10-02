// RUN: iree-opt %s \
// RUN:   --iree-abi-transformation-pipeline \
// RUN:   --iree-flow-transformation-pipeline \
// RUN:   --iree-preloaded-transforms=%p/transform_spec.mlir

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
