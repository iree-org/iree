// RUN: iree-compile --iree-hal-target-backends=vmvx %s | iree-run-module --module=- --function=expect_true_of_false | FileCheck %s

// Tests that even if the check module is not available (in this case because
// we are running with iree-run-module instead of iree-check-module) the
// execution still completes.

// CHECK-LABEL: EXEC @expect_true_of_false
// CHECK: result[0]: i32=0
module @expect_failure {
  func.func @expect_true_of_false() -> i32 {
    %false = util.unfoldable_constant 0 : i32
    check.expect_true(%false) : i32
    return %false : i32
  }
}
