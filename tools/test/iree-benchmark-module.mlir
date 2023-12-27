// RUN: iree-compile --iree-hal-target-backends=vmvx %s | iree-benchmark-module --device=local-task --module=- --function=abs --input=f32=-2 | FileCheck %s
// RUN: iree-compile --iree-hal-target-backends=llvm-cpu %s | iree-benchmark-module --device=local-task --module=- --function=abs --input=f32=-2 | FileCheck %s

// CHECK-LABEL: BM_abs
func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
  %result = math.absf %input : tensor<f32>
  return %result : tensor<f32>
}
