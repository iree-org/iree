// RUN: (iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:  iree-run-module --module=- --function=many_tensor) | \
// RUN:  FileCheck %s
// RUN: (iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:  iree-benchmark-module --device=local-task --module=- --function=many_tensor) | \
// RUN:  FileCheck --check-prefix=BENCHMARK %s
// RUN: (iree-run-mlir --Xcompiler,iree-hal-target-device=local -Xcompiler,iree-hal-local-target-device-backends=vmvx %s) | \
// RUN:  FileCheck %s

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
