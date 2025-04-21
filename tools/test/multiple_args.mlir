// RUN: iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:   iree-run-module --module=- --function=multi_input --input="2xi32=[1 2]" --input="2xi32=[3 4]" | \
// RUN:   FileCheck %s
// RUN: iree-run-mlir \
// RUN:   --Xcompiler,iree-hal-target-device=local \
// RUN:   --Xcompiler,iree-hal-local-target-device-backends=vmvx \
// RUN:   %s \
// RUN:   --input="2xi32=[1 2]" --input="2xi32=[3 4]" | \
// RUN:   FileCheck %s
// RUN: iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:   iree-benchmark-module --device=local-task --module=- --function=multi_input --input="2xi32=[1 2]" --input="2xi32=[3 4]" | \
// RUN:   FileCheck --check-prefix=BENCHMARK %s

// BENCHMARK-LABEL: BM_multi_input
// CHECK-LABEL: EXEC @multi_input
func.func @multi_input(%arg0 : tensor<2xi32>, %arg1 : tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>) {
  return %arg0, %arg1 : tensor<2xi32>, tensor<2xi32>
}
// CHECK: 2xi32=1 2
// CHECK: 2xi32=3 4
