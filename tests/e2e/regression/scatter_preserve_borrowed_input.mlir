// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=scatter_preserves_input --input="8xi32=[0 1 2 3 4 5 6 7]" --input="3xi32=[100 200 300]" | FileCheck %s --check-prefix=SCATTER

func.func @scatter_preserves_input(%original: tensor<8xi32>,
                                   %updates: tensor<3xi32>)
    -> (tensor<8xi32>, tensor<8xi32>) {
  %indices = arith.constant dense<[[1], [3], [5]]> : tensor<3x1xi32>
  %result = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
      ins(%updates, %indices : tensor<3xi32>, tensor<3x1xi32>)
      outs(%original : tensor<8xi32>) {
  ^bb0(%update: i32, %old: i32):
    iree_linalg_ext.yield %update : i32
  } -> tensor<8xi32>
  return %result, %original : tensor<8xi32>, tensor<8xi32>
}
// SCATTER-LABEL: EXEC @scatter_preserves_input
// SCATTER: 8xi32=0 100 2 200 4 300 6 7
// SCATTER: 8xi32=0 1 2 3 4 5 6 7
