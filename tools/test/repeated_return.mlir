// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | iree-run-module --module=- --function=many_tensor) | FileCheck %s
// RUN: iree-compile --iree-hal-target-backends=vmvx %s | iree-benchmark-module --device=local-task --module=- --function=many_tensor | FileCheck --check-prefix=BENCHMARK %s
// RUN: iree-run-mlir --iree-hal-target-backends=vmvx %s | FileCheck %s

// BENCHMARK-LABEL: BM_many_tensor
// CHECK-LABEL: EXEC @many_tensor
func.func @many_tensor() -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>,
                        tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) {
  %res = util.unfoldable_constant
      dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  return %res, %res, %res, %res, %res, %res :
        tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>,
        tensor<2x2xf32>, tensor<2x2xf32>
}
// CHECK-COUNT-6: 2x2xf32=[1 2][3 4]
