// RUN: iree-opt --iree-common-input-transformation-pipeline %s | \
// RUN: iree-opt --iree-abi-transformation-pipeline - | \
// RUN: iree-opt --iree-common-input-transformation-pipeline - | \
// RUN: iree-opt --iree-global-optimization-transformation-pipeline - | \
// RUN: iree-opt --iree-flow-transformation-pipeline - | \
// RUN: iree-opt --iree-stream-transformation-pipeline - | \
// RUN: iree-opt --iree-hal-transformation-pipeline --iree-hal-target-backends=vmvx - | \
// RUN: iree-opt --iree-vm-transformation-pipeline - | \
// RUN: FileCheck %s

// CHECK: vm.func private @abs(%arg0: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view>
func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
  %result = math.absf %input : tensor<f32>
  return %result : tensor<f32>
}
