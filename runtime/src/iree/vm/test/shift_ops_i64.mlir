vm.module @shift_ops_i64 {

  //===--------------------------------------------------------------------===//
  // ExtI64: Native bitwise shifts and rotates
  //===--------------------------------------------------------------------===//

  vm.export @test_shl_i64
  vm.func @test_shl_i64() {
    %c1 = vm.const.i64 1
    %c1dno = util.optimization_barrier %c1 : i64
    %shamt = vm.const.i32 2
    %v = vm.shl.i64 %c1dno, %shamt : i64
    %c4 = vm.const.i64 4
    vm.check.eq %v, %c4, "1<<2=4" : i64
    vm.return
  }

  vm.export @test_shr_i64s
  vm.func @test_shr_i64s() {
    %c1 = vm.const.i64 -1
    %c1dno = util.optimization_barrier %c1 : i64
    %shamt = vm.const.i32 2
    %v = vm.shr.i64.s %c1dno, %shamt : i64
    %cn1 = vm.const.i64 -1
    vm.check.eq %v, %cn1, "-1>>2=-1" : i64
    vm.return
  }

  vm.export @test_shr_i64u
  vm.func @test_shr_i64u() {
    %c4 = vm.const.i64 4
    %c4dno = util.optimization_barrier %c4 : i64
    %shamt = vm.const.i32 2
    %v = vm.shr.i64.u %c4dno, %shamt : i64
    %c1 = vm.const.i64 1
    vm.check.eq %v, %c1, "4>>2=1" : i64
    vm.return
  }
}
