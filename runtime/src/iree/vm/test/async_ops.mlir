// Tested by iree/vm/bytecode/dispatch_async_test.cc.
//
// NOTE: we don't want to rely on vm.check.* and the main runner here for
// testing as it makes it hard to test failure cases; a test that doesn't run
// because we don't resume from the caller would look like a success. The test
// runner has the other half of this code with the expectations.

vm.module @async_ops {

  //===--------------------------------------------------------------------===//
  // vm.yield
  //===--------------------------------------------------------------------===//

  // Tests a simple straight-line yield sequence that requires 3 resumes.
  //
  // Expects a result of %arg0 + 3.
  vm.export @yield_sequence
  vm.func @yield_sequence(%arg0: i32) -> i32 {
    %c1 = vm.const.i32 1
    %y0 = vm.add.i32 %arg0, %c1 : i32
    %y0_dno = util.optimization_barrier %y0 : i32
    vm.yield ^bb1
  ^bb1:
    %y1 = vm.add.i32 %y0_dno, %c1 : i32
    %y1_dno = util.optimization_barrier %y1 : i32
    vm.yield ^bb2
  ^bb2:
    %y2 = vm.add.i32 %y1_dno, %c1 : i32
    %y2_dno = util.optimization_barrier %y2 : i32
    vm.yield ^bb3
  ^bb3:
    vm.return %y2_dno : i32
  }

  // Tests a yield with data-dependent control, ensuring that we run the
  // alternating branches and pass along branch args on resume.
  //
  // Expects a result of %arg0 ? %arg1 : %arg2.
  vm.export @yield_divergent
  vm.func @yield_divergent(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    %cond = vm.cmp.nz.i32 %arg0 : i32
    vm.cond_br %cond, ^true, ^false
  ^true:
    %arg1_dno = util.optimization_barrier %arg1 : i32
    vm.yield ^bb3(%arg1_dno : i32)
  ^false:
    %arg2_dno = util.optimization_barrier %arg2 : i32
    vm.yield ^bb3(%arg2_dno: i32)
  ^bb3(%result : i32):
    vm.return %result : i32
  }

}
