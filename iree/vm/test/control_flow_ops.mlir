vm.module @control_flow_ops {

  //===--------------------------------------------------------------------===//
  // vm.return
  //===--------------------------------------------------------------------===//

  vm.export @test_return_empty
  vm.func @test_return_empty() {
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.fail
  //===--------------------------------------------------------------------===//

  vm.export @fail_always
  vm.func @fail_always() {
    %code = vm.const.i32 4 : i32
    vm.fail %code, "error!"
  }

  //===--------------------------------------------------------------------===//
  // vm.check.*
  //===--------------------------------------------------------------------===//

  vm.export @test_check_eq_always
  vm.func @test_check_eq_always() {
    %c1 = vm.const.i32 1 : i32
    %c1dno = util.do_not_optimize(%c1) : i32
    vm.check.eq %c1, %c1dno, "error!" : i32
    vm.return
  }

  vm.export @fail_check_eq_never
  vm.func @fail_check_eq_never() {
    %c1 = vm.const.i32 1 : i32
    %c2 = vm.const.i32 2 : i32
    %c1dno = util.do_not_optimize(%c1) : i32
    %c2dno = util.do_not_optimize(%c2) : i32
    vm.check.eq %c1dno, %c2dno, "error!" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.cond_br
  //===--------------------------------------------------------------------===//

  vm.export @test_cond_br
  vm.func @test_cond_br() {
    %c1 = vm.const.i32 1 : i32
    %c1dno = util.do_not_optimize(%c1) : i32
    vm.cond_br %c1dno, ^bb1, ^bb2
  ^bb1:
    vm.check.eq %c1dno, %c1dno, "error!" : i32
    vm.return
  ^bb2:
    %code = vm.const.i32 4 : i32
    vm.fail %code, "unreachable!"
  }

  vm.export @test_cond_br_int_arg
  vm.func @test_cond_br_int_arg() {
    %c1 = vm.const.i32 1 : i32
    %c1dno = util.do_not_optimize(%c1) : i32
    vm.cond_br %c1dno, ^bb1(%c1dno : i32), ^bb2(%c1dno : i32)
  ^bb1(%arg1 : i32):
    vm.check.eq %arg1, %c1dno, "error!" : i32
    vm.return
  ^bb2(%arg2 : i32):
    %code = vm.const.i32 4 : i32
    vm.fail %code, "unreachable!"
  }

  // TODO(#7487): Enable the test for emitc.
  vm.export @test_cond_br_ref_arg attributes {emitc.exclude}
  vm.func private @test_cond_br_ref_arg() {
    %c1 = vm.const.i32 1 : i32
    %c1dno = util.do_not_optimize(%c1) : i32
    %ref = vm.const.ref.zero : !vm.ref<?>
    vm.cond_br %c1dno, ^bb1(%ref : !vm.ref<?>), ^bb2(%ref : !vm.ref<?>)
  ^bb1(%arg1 : !vm.ref<?>):
    vm.check.eq %arg1, %ref, "error!" : !vm.ref<?>
    vm.return
  ^bb2(%arg2 : !vm.ref<?>):
    %code = vm.const.i32 4 : i32
    vm.fail %code, "unreachable!"
  }

}
