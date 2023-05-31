// RUN: iree-compile --iree-hal-target-backends=vmvx %s | iree-run-module --module=- --function=multi_input --input="2xi32=[1 2]" --input="2xi32=[3 4]" | FileCheck %s
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=vmvx %s --input="2xi32=[1 2]" --input="2xi32=[3 4]" | FileCheck %s
// RUN: iree-compile --iree-hal-target-backends=vmvx %s | iree-benchmark-module --device=local-task --module=- --function=multi_input --input="2xi32=[1 2]" --input="2xi32=[3 4]" | FileCheck --check-prefix=BENCHMARK %s

// BENCHMARK-LABEL: BM_multi_input
// CHECK-LABEL: EXEC @multi_input
func.func @multi_input(%arg0 : tensor<2xi32>, %arg1 : tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>) {
  return %arg0, %arg1 : tensor<2xi32>, tensor<2xi32>
}
// CHECK: 2xi32=1 2
// CHECK: 2xi32=3 4
