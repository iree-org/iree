vm.module @shift_ops {

  //===--------------------------------------------------------------------===//
  // Native bitwise shifts and rotates
  //===--------------------------------------------------------------------===//

  vm.export @test_shl_i32
  vm.func @test_shl_i32() {
    %c1 = vm.const.i32 1
    %c1dno = util.optimization_barrier %c1 : i32
    %c2 = vm.const.i32 2
    %v = vm.shl.i32 %c1dno, %c2 : i32
    %c4 = vm.const.i32 4
    vm.check.eq %v, %c4, "1<<2=4" : i32
    vm.return
  }

  vm.export @test_shr_i32s
  vm.func @test_shr_i32s() {
    %cn1 = vm.const.i32 -1
    %cn1dno = util.optimization_barrier %cn1 : i32
    %c2 = vm.const.i32 2
    %v = vm.shr.i32.s %cn1dno, %c2 : i32
    vm.check.eq %v, %cn1dno, "-1>>2=-1" : i32
    vm.return
  }

  vm.export @test_shr_i32u
  vm.func @test_shr_i32u() {
    %c4 = vm.const.i32 4
    %c4dno = util.optimization_barrier %c4 : i32
    %c2 = vm.const.i32 2
    %v = vm.shr.i32.u %c4dno, %c2 : i32
    %c1 = vm.const.i32 1
    vm.check.eq %v, %c1, "4>>2=1" : i32
    vm.return
  }
}
