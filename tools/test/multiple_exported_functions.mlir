// RUN: iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:   iree-benchmark-module --device=local-task --module=- | \
// RUN:   FileCheck %s

module {
  func.func @foo1() -> tensor<4xf32> {
    %input = util.unfoldable_constant dense<[0.0, 1.0, 2.0, 4.0]> : tensor<4xf32>
    %result = math.exp %input : tensor<4xf32>
    return %result : tensor<4xf32>
  }
  func.func @foo2() -> tensor<4xf32> {
    %input = util.unfoldable_constant dense<[0.0, 1.0, 2.0, 4.0]> : tensor<4xf32>
    %result = math.absf %input : tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
// CHECK: BM_foo1
// CHECK: BM_foo2
