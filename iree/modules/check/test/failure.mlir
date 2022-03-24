// RUN: iree-compile --iree-input-type=mhlo --iree-hal-target-backends=vmvx -iree-mlir-to-vm-bytecode-module %s | iree-check-module --expect_failure - | FileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-compile --iree-input-type=mhlo --iree-hal-target-backends=vulkan-spirv -iree-mlir-to-vm-bytecode-module %s | iree-check-module --driver=vulkan --expect_failure - | FileCheck %s)

// CHECK-LABEL: expect_failure.expect_true_of_false
// CHECK: Expected 0 to be nonzero
// CHECK: Test failed as expected
module @expect_failure {
func @expect_true_of_false() {
  %false = util.unfoldable_constant 0 : i32
  check.expect_true(%false) : i32
  return
}
}
