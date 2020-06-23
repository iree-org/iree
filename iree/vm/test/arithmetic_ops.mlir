vm.module @arithmetic_ops {

  //===--------------------------------------------------------------------===//
  // I32 Arithmetic
  //===--------------------------------------------------------------------===//

  vm.export @test_add_i32
  vm.func @test_add_i32() {
    %c1 = vm.const.i32 1 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %v = vm.add.i32 %c1dno, %c1dno : i32
    %c2 = vm.const.i32 2 : i32
    vm.check.eq %v, %c2, "1+1=2" : i32
    vm.return
  }

}
