// Executes an inline HAL (no device) module via the hal_loader module.
// RUN: (iree-compile --iree-execution-model=inline-dynamic --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:  iree-run-module --module=- --function=abs --input="2xf32=-2 3") | FileCheck %s
// RUN: (iree-compile --iree-execution-model=inline-dynamic --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu %s | \
// RUN:  iree-run-module --module=- --function=abs --input="2xf32=-2 3") | FileCheck %s

// CHECK-LABEL: EXEC @abs
func.func @abs(%input : tensor<2xf32>) -> (tensor<2xf32>) {
  %result = math.absf %input : tensor<2xf32>
  return %result : tensor<2xf32>
}
// CHECK: result[0]: hal.buffer_view
// CHECK-NEXT: 2xf32=2 3
