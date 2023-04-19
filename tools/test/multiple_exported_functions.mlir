// RUN: iree-compile --iree-hal-target-backends=vmvx %s | iree-benchmark-module --device=local-task --module=- | FileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-compile --iree-hal-target-backends=vulkan-spirv %s | iree-benchmark-module --device=vulkan --module=- | FileCheck %s)

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
