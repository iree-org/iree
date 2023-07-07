// RUN: iree-compile --iree-hal-target-backends=vmvx %s | iree-check-module --module=- --expect_failure | FileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-compile --iree-hal-target-backends=vulkan-spirv %s | iree-check-module --device=vulkan --module=- --expect_failure | FileCheck %s)

// CHECK-LABEL: expect_failure.expect_true_of_false
// CHECK: Expected 0 to be nonzero
// CHECK: Test failed as expected
module @expect_failure {
func.func @expect_true_of_false() {
  %false = util.unfoldable_constant 0 : i32
  check.expect_true(%false) : i32
  return
}
}
