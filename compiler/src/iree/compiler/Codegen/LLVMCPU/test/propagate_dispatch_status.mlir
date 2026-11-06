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

// -----

// Failed calls return poison, so result attributes leave rewritten calls as
// they leave the helpers; argument attributes and other metadata stay.
module {
  func.func private @scaled(%status: i32 {test.arg = "status"}, %cond: i1) -> (i32 {test.tag = "value"}) {
    scf.if %cond {
      util.status.check_ok %status
    }
    %result = arith.constant 42 : i32
    return %result : i32
  }

  func.func private @checked(%status: i32) {
    util.status.check_ok %status
    return
  }

  func.func public @entry_point() {
    %status = arith.constant 9 : i32
    %true = arith.constant true
    %result = "func.call"(%status, %true) <{callee = @scaled, arg_attrs = [{test.arg = "status"}, {}], res_attrs = [{test.tag = "value"}]}> {test.meta} : (i32, i1) -> i32
    "func.call"(%status) <{callee = @checked, res_attrs = []}> : (i32) -> ()
    return
  }
}
// CHECK-LABEL: func.func private @scaled(
// CHECK-SAME:    %{{.+}}: i32 {test.arg = "status"}, %{{.+}}: i1) -> (i32, i32)
// CHECK-LABEL: func.func public @entry_point()
// CHECK:         call @scaled(%{{.+}}, %{{.+}}) <arg_attrs = [{test.arg = "status"}, {}]> {test.meta} : (i32, i1) -> (i32, i32)
// CHECK:         call @checked(%{{.+}}) : (i32) -> i32
