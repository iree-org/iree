// RUN: check-translate --iree-hal-target-backends=interpreter-bytecode -iree-mlir-to-vm-bytecode-module %s | iree-check-module --expect_failure
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (check-translate --iree-hal-target-backends=vulkan-spirv -iree-mlir-to-vm-bytecode-module %s | iree-check-module --driver=vulkan --expect_failure)

func @check_false() attributes { iree.module.export } {
  %false = iree.unfoldable_constant 0 : i32
  check.expect_true(%false) : i32
  return
}
