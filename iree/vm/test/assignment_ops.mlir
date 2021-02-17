vm.module @assignment_ops {

  //===--------------------------------------------------------------------===//
  // Conditional assignment
  //===--------------------------------------------------------------------===//

  // TODO: The CModuleTarget enforces exports to be ordered.
  vm.export @test_select_i32

  vm.func @test_select_i32() {
    %c0 = vm.const.i32 0 : i32
    %c0dno = iree.do_not_optimize(%c0) : i32
    %c1 = vm.const.i32 1 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %v1 = vm.select.i32 %c0dno, %c0dno, %c1dno : i32
    vm.check.eq %v1, %c1, "0 ? 0 : 1 = 1" : i32
    %v2 = vm.select.i32 %c1dno, %c0dno, %c1dno : i32
    vm.check.eq %v2, %c0, "1 ? 0 : 1 = 0" : i32
    vm.return
  }
}
