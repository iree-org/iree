vm.module @arithmetic_ops_i64 {

  //===--------------------------------------------------------------------===//
  // I64 Arithmetic
  //===--------------------------------------------------------------------===//

  vm.export @test_add_i64
  vm.func @test_add_i64() {
    %c1 = vm.const.i64 1 : i64
    %c1dno = iree.do_not_optimize(%c1) : i64
    %v = vm.add.i64 %c1dno, %c1dno : i64
    %c2 = vm.const.i64 2 : i64
    vm.check.eq %v, %c2, "1+1=2" : i64
    vm.return
  }

}
