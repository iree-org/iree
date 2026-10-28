// RUN: iree-opt --iree-llvmcpu-propagate-dispatch-status --split-input-file %s | FileCheck %s

// The failing helper and its caller thread an i32 status; only they lose their
// structured control flow.
module {
  func.func private @helper(%status: i32, %cond: i1) -> i32 {
    scf.if %cond {
      util.status.check_ok %status
    }
    %result = arith.constant 42 : i32
    return %result : i32
  }

  func.func private @untouched(%lb: index, %ub: index, %step: index) {
    scf.for %i = %lb to %ub step %step {
    }
    return
  }

  func.func public @entry_point() {
    %status = arith.constant 9 : i32
    %true = arith.constant true
    %result = func.call @helper(%status, %true) : (i32, i1) -> i32
    return
  }
}
// CHECK-LABEL: func.func private @helper(
// CHECK-SAME:    -> (i32, i32)
// CHECK-NOT:     scf.if
// CHECK:         cf.cond_br
// CHECK:         return %{{.+}}, %{{.+}} : i32, i32
// CHECK-LABEL: func.func private @untouched(
// CHECK:         scf.for
// CHECK-LABEL: func.func public @entry_point()
// CHECK-SAME:    -> i32
// CHECK-SAME:    iree.returns_status
// CHECK:         %[[CALL:.+]]:2 = call @helper(
// CHECK:         cf.cond_br
