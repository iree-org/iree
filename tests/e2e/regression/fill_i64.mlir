// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu %s --input=2x3xi64 | FileCheck %s
// RUN: [[ $IREE_VMVX_DISABLE == 1 ]] || (iree-run-mlir --Xcompiler,iree-hal-target-backends=vmvx %s --input=2x3xi64 | FileCheck %s)

// CHECK: EXEC @fill_i64
func.func @fill_i64(%arg0: tensor<?x?xi64>) -> (tensor<?x?xi64>, tensor<?x?xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xi64>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xi64>

  %cv0 = arith.constant -1 : i64
  %v0_init = tensor.empty(%0, %1) : tensor<?x?xi64>
  %v0 = linalg.fill ins(%cv0 : i64) outs(%v0_init : tensor<?x?xi64>) -> tensor<?x?xi64>
  // CHECK: 2x3xi64=[-1 -1 -1][-1 -1 -1]

  %cv1 = arith.constant 9223372036854775807 : i64
  %v1_init = tensor.empty(%0, %1) : tensor<?x?xi64>
  %v1 = linalg.fill ins(%cv1 : i64) outs(%v1_init : tensor<?x?xi64>) -> tensor<?x?xi64>
  // CHECK: 2x3xi64=[9223372036854775807 9223372036854775807 9223372036854775807][9223372036854775807 9223372036854775807 9223372036854775807]

  return %v0, %v1 : tensor<?x?xi64>, tensor<?x?xi64>
}
