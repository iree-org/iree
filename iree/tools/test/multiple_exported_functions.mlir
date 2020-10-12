// RUN: iree-translate --iree-hal-target-backends=vmla -iree-mlir-to-vm-bytecode-module %s -o ${TEST_TMPDIR?}/bc.module && iree-benchmark-module --driver=vmla --module_file=${TEST_TMPDIR?}/bc.module | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-translate --iree-hal-target-backends=vulkan-spirv -iree-mlir-to-vm-bytecode-module %s -o ${TEST_TMPDIR?}/bc.module && iree-benchmark-module --driver=vulkan --module_file=${TEST_TMPDIR?}/bc.module | IreeFileCheck %s)

module {
  func @foo1() -> tensor<4xf32> attributes { iree.module.export } {
    %input = iree.unfoldable_constant dense<[0.0, 1.0, 2.0, 4.0]> : tensor<4xf32>
    %result = "mhlo.exponential"(%input) : (tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
  func @foo2() -> tensor<4xf32> attributes { iree.module.export } {
    %input = iree.unfoldable_constant dense<[0.0, 1.0, 2.0, 4.0]> : tensor<4xf32>
    %result = "mhlo.abs"(%input) : (tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
// CHECK: BM_foo1
// CHECK: BM_foo2
