vm.module @shift_ops {

  //===--------------------------------------------------------------------===//
  // Native bitwise shifts and rotates
  //===--------------------------------------------------------------------===//

  vm.export @test_shl_i32
  vm.func @test_shl_i32() {
    %c1 = vm.const.i32 1 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %v = vm.shl.i32 %c1dno, 2 : i32
    %c2 = vm.const.i32 4 : i32
    vm.check.eq %v, %c2, "1<<2=4" : i32
    vm.return
  }

  vm.export @test_shr_i32s
  vm.func @test_shr_i32s() {
    %c1 = vm.const.i32 -1 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %v = vm.shr.i32.s %c1dno, 2 : i32
    %c2 = vm.const.i32 -1 : i32
    vm.check.eq %v, %c2, "-1>>-1=-1" : i32
    vm.return
  }

  vm.export @test_shr_i32u
  vm.func @test_shr_i32u() {
    %c1 = vm.const.i32 4 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %v = vm.shr.i32.u %c1dno, 2 : i32
    %c2 = vm.const.i32 1 : i32
    vm.check.eq %v, %c2, "4>>2=1" : i32
    vm.return
  }
}
