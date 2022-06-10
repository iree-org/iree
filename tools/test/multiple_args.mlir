// RUN: iree-compile --iree-hal-target-backends=vmvx --iree-mlir-to-vm-bytecode-module %s | iree-run-module --entry_function=multi_input --function_input="2xi32=[1 2]" --function_input="2xi32=[3 4]" | FileCheck %s
// RUN: iree-run-mlir --iree-hal-target-backends=vmvx --function-input='2xi32=[1 2]' --function-input='2xi32=[3 4]' %s | FileCheck %s
// RUN: iree-compile --iree-hal-target-backends=vmvx --iree-mlir-to-vm-bytecode-module %s | iree-benchmark-module --device=local-task --entry_function=multi_input --function_input="2xi32=[1 2]" --function_input="2xi32=[3 4]" | FileCheck --check-prefix=BENCHMARK %s

// BENCHMARK-LABEL: BM_multi_input
// CHECK-LABEL: EXEC @multi_input
func.func @multi_input(%arg0 : tensor<2xi32>, %arg1 : tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>) {
  return %arg0, %arg1 : tensor<2xi32>, tensor<2xi32>
}
// CHECK: 2xi32=1 2
// CHECK: 2xi32=3 4
