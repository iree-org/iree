// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu %s | FileCheck %s
// RUN: [[ $IREE_VMVX_DISABLE == 1 ]] || (iree-run-mlir --Xcompiler,iree-hal-target-backends=vmvx %s | FileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --Xcompiler,iree-hal-target-backends=vulkan-spirv %s | FileCheck %s)

func.func @tensor_cast() -> tensor<2x?xf32> {
  %input = util.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %result = tensor.cast %input : tensor<2x3xf32> to tensor<2x?xf32>
  return %result : tensor<2x?xf32>
}
// CHECK: 2x3xf32=[1 2 3][4 5 6]
