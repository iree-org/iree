vm.module @shift_ops_i64 {

  //===--------------------------------------------------------------------===//
  // ExtI64: Native bitwise shifts and rotates
  //===--------------------------------------------------------------------===//

  vm.export @test_shl_i64
  vm.func @test_shl_i64() {
    %c1 = vm.const.i64 1 : i64
    %c1dno = iree.do_not_optimize(%c1) : i64
    %v = vm.shl.i64 %c1dno, 2 : i64
    %c2 = vm.const.i64 4 : i64
    vm.check.eq %v, %c2, "1<<2=4" : i64
    vm.return
  }

  vm.export @test_shr_i64s
  vm.func @test_shr_i64s() {
    %c1 = vm.const.i64 -1 : i64
    %c1dno = iree.do_not_optimize(%c1) : i64
    %v = vm.shr.i64.s %c1dno, 2 : i64
    %c2 = vm.const.i64 -1 : i64
    vm.check.eq %v, %c2, "-1>>-1=-1" : i64
    vm.return
  }

  vm.export @test_shr_i64u
  vm.func @test_shr_i64u() {
    %c1 = vm.const.i64 4 : i64
    %c1dno = iree.do_not_optimize(%c1) : i64
    %v = vm.shr.i64.u %c1dno, 2 : i64
    %c2 = vm.const.i64 1 : i64
    vm.check.eq %v, %c2, "4>>2=1" : i64
    vm.return
  }
}
