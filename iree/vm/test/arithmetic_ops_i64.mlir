vm.module @arithmetic_ops_i64 {

  //===--------------------------------------------------------------------===//
  // ExtI64: Native integer arithmetic
  //===--------------------------------------------------------------------===//

  vm.export @test_add_i64
  vm.func @test_add_i64() {
    %c1 = vm.const.i64 1
    %c1dno = util.do_not_optimize(%c1) : i64
    %v = vm.add.i64 %c1dno, %c1dno : i64
    %c2 = vm.const.i64 2
    vm.check.eq %v, %c2, "1+1=2" : i64
    vm.return
  }

  vm.export @test_sub_i64
  vm.func @test_sub_i64() {
    %c1 = vm.const.i64 3
    %c1dno = util.do_not_optimize(%c1) : i64
    %c2 = vm.const.i64 2
    %c2dno = util.do_not_optimize(%c2) : i64
    %v = vm.sub.i64 %c1dno, %c2dno : i64
    %c3 = vm.const.i64 1
    vm.check.eq %v, %c3, "3-2=1" : i64
    vm.return
  }

  vm.export @test_mul_i64
  vm.func @test_mul_i64() {
    %c1 = vm.const.i64 2
    %c1dno = util.do_not_optimize(%c1) : i64
    %v = vm.mul.i64 %c1dno, %c1dno : i64
    %c2 = vm.const.i64 4
    vm.check.eq %v, %c2, "2*2=4" : i64
    vm.return
  }

  vm.export @test_div_i64s
  vm.func @test_div_i64s() {
    %c1 = vm.const.i64 4
    %c1dno = util.do_not_optimize(%c1) : i64
    %c2 = vm.const.i64 -2
    %c2dno = util.do_not_optimize(%c2) : i64
    %v = vm.div.i64.s %c1dno, %c2dno : i64
    %c3 = vm.const.i64 -2
    vm.check.eq %v, %c3, "4/-2=-2" : i64
    vm.return
  }

  vm.export @test_div_i64u
  vm.func @test_div_i64u() {
    %c1 = vm.const.i64 4
    %c1dno = util.do_not_optimize(%c1) : i64
    %c2 = vm.const.i64 2
    %c2dno = util.do_not_optimize(%c2) : i64
    %v = vm.div.i64.u %c1dno, %c2dno : i64
    %c3 = vm.const.i64 2
    vm.check.eq %v, %c3, "4/2=2" : i64
    vm.return
  }

  vm.export @test_rem_i64s
  vm.func @test_rem_i64s() {
    %c1 = vm.const.i64 -3
    %c1dno = util.do_not_optimize(%c1) : i64
    %c2 = vm.const.i64 -2
    %c2dno = util.do_not_optimize(%c2) : i64
    %v = vm.rem.i64.s %c1dno, %c2dno : i64
    %c3 = vm.const.i64 -1
    vm.check.eq %v, %c3, "-3%-2=-1" : i64
    vm.return
  }

  vm.export @test_rem_i64u
  vm.func @test_rem_i64u() {
    %c1 = vm.const.i64 3
    %c1dno = util.do_not_optimize(%c1) : i64
    %c2 = vm.const.i64 2
    %c2dno = util.do_not_optimize(%c2) : i64
    %v = vm.rem.i64.u %c1dno, %c2dno : i64
    %c3 = vm.const.i64 1
    vm.check.eq %v, %c3, "3%2=1" : i64
    vm.return
  }

  vm.export @test_fma_i64
  vm.func @test_fma_i64() {
    %c2 = vm.const.i64 2
    %c2dno = util.do_not_optimize(%c2) : i64
    %c3 = vm.const.i64 3
    %c3dno = util.do_not_optimize(%c3) : i64
    %c5 = vm.const.i64 5
    %c5dno = util.do_not_optimize(%c5) : i64
    %v = vm.fma.i64 %c2dno, %c3dno, %c5dno : i64
    %c11 = vm.const.i64 11
    vm.check.eq %v, %c11, "2*3+5=11" : i64
    vm.return
  }

  vm.export @test_not_i64
  vm.func @test_not_i64() {
    %c1 = vm.const.i64 0
    %c1dno = util.do_not_optimize(%c1) : i64
    %v = vm.not.i64 %c1dno : i64
    %c2 = vm.const.i64 -1
    vm.check.eq %v, %c2, "~0=-1" : i64
    vm.return
  }

  vm.export @test_and_i64
  vm.func @test_and_i64() {
    %c1 = vm.const.i64 5
    %c1dno = util.do_not_optimize(%c1) : i64
    %c2 = vm.const.i64 3
    %c2dno = util.do_not_optimize(%c2) : i64
    %v = vm.and.i64 %c1dno, %c2dno : i64
    %c3 = vm.const.i64 1
    vm.check.eq %v, %c3, "5&3=1" : i64
    vm.return
  }

  vm.export @test_or_i64
  vm.func @test_or_i64() {
    %c1 = vm.const.i64 5
    %c1dno = util.do_not_optimize(%c1) : i64
    %c2 = vm.const.i64 3
    %c2dno = util.do_not_optimize(%c2) : i64
    %v = vm.or.i64 %c1dno, %c2dno : i64
    %c3 = vm.const.i64 7
    vm.check.eq %v, %c3, "5|3=7" : i64
    vm.return
  }

  vm.export @test_xor_i64
  vm.func @test_xor_i64() {
    %c1 = vm.const.i64 5
    %c1dno = util.do_not_optimize(%c1) : i64
    %c2 = vm.const.i64 3
    %c2dno = util.do_not_optimize(%c2) : i64
    %v = vm.xor.i64 %c1dno, %c2dno : i64
    %c3 = vm.const.i64 6
    vm.check.eq %v, %c3, "5^3=6" : i64
    vm.return
  }
}
