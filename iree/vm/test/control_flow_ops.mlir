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
    %c1dno = iree.do_not_optimize(%c1) : i32
    vm.check.eq %c1, %c1dno, "error!" : i32
    vm.return
  }

  vm.export @fail_check_eq_never
  vm.func @fail_check_eq_never() {
    %c1 = vm.const.i32 1 : i32
    %c2 = vm.const.i32 2 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %c2dno = iree.do_not_optimize(%c2) : i32
    vm.check.eq %c1dno, %c2dno, "error!" : i32
    vm.return
  }

}
