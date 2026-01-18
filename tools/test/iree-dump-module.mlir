// RUN: iree-compile \
// RUN:   %s \
// RUN:   --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   -o=%t.vmfb && \
// RUN: iree-dump-module \
// RUN:   --output=all \
// RUN:   %t.vmfb | \
// RUN: FileCheck %s

// CHECK: @module : version 0

// CHECK: fn0
func.func @fn0(%input : tensor<f32>) -> (tensor<f32>) {
  // CHECK: [{{[0-9]+}}]{{.*}}<block>
  %result = math.absf %input : tensor<f32>
  return %result : tensor<f32>
}

// CHECK: fn1
func.func @fn1(%input : tensor<f32>) -> (tensor<f32>) {
  // CHECK: [{{[0-9]+}}]{{.*}}<block>
  %result = arith.mulf %input, %input : tensor<f32>
  return %result : tensor<f32>
}

// CHECK: __init
