vm.module @arithmetic_ops_i64 {

  //===--------------------------------------------------------------------===//
  // ExtI64: Native integer arithmetic
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

  vm.export @test_sub_i64
  vm.func @test_sub_i64() {
    %c1 = vm.const.i64 3 : i64
    %c1dno = iree.do_not_optimize(%c1) : i64
    %c2 = vm.const.i64 2 : i64
    %c2dno = iree.do_not_optimize(%c2) : i64
    %v = vm.sub.i64 %c1dno, %c2dno : i64
    %c3 = vm.const.i64 1 : i64
    vm.check.eq %v, %c3, "3-2=1" : i64
    vm.return
  }

  vm.export @test_mul_i64
  vm.func @test_mul_i64() {
    %c1 = vm.const.i64 2 : i64
    %c1dno = iree.do_not_optimize(%c1) : i64
    %v = vm.mul.i64 %c1dno, %c1dno : i64
    %c2 = vm.const.i64 4 : i64
    vm.check.eq %v, %c2, "2*2=4" : i64
    vm.return
  }

  vm.export @test_div_i64s
  vm.func @test_div_i64s() {
    %c1 = vm.const.i64 4 : i64
    %c1dno = iree.do_not_optimize(%c1) : i64
    %c2 = vm.const.i64 -2 : i64
    %c2dno = iree.do_not_optimize(%c2) : i64
    %v = vm.div.i64.s %c1dno, %c2dno : i64
    %c3 = vm.const.i64 -2 : i64
    vm.check.eq %v, %c3, "4/-2=-2" : i64
    vm.return
  }

  vm.export @test_div_i64u
  vm.func @test_div_i64u() {
    %c1 = vm.const.i64 4 : i64
    %c1dno = iree.do_not_optimize(%c1) : i64
    %c2 = vm.const.i64 2 : i64
    %c2dno = iree.do_not_optimize(%c2) : i64
    %v = vm.div.i64.u %c1dno, %c2dno : i64
    %c3 = vm.const.i64 2 : i64
    vm.check.eq %v, %c3, "4/2=2" : i64
    vm.return
  }

  vm.export @test_rem_i64s
  vm.func @test_rem_i64s() {
    %c1 = vm.const.i64 -3 : i64
    %c1dno = iree.do_not_optimize(%c1) : i64
    %c2 = vm.const.i64 -2 : i64
    %c2dno = iree.do_not_optimize(%c2) : i64
    %v = vm.rem.i64.s %c1dno, %c2dno : i64
    %c3 = vm.const.i64 -1 : i64
    vm.check.eq %v, %c3, "-3%-2=-1" : i64
    vm.return
  }

  vm.export @test_rem_i64u
  vm.func @test_rem_i64u() {
    %c1 = vm.const.i64 3 : i64
    %c1dno = iree.do_not_optimize(%c1) : i64
    %c2 = vm.const.i64 2 : i64
    %c2dno = iree.do_not_optimize(%c2) : i64
    %v = vm.rem.i64.u %c1dno, %c2dno : i64
    %c3 = vm.const.i64 1 : i64
    vm.check.eq %v, %c3, "3%2=1" : i64
    vm.return
  }

  vm.export @test_not_i64
  vm.func @test_not_i64() {
    %c1 = vm.const.i64 0 : i64
    %c1dno = iree.do_not_optimize(%c1) : i64
    %v = vm.not.i64 %c1dno : i64
    %c2 = vm.const.i64 -1 : i64
    vm.check.eq %v, %c2, "~0=-1" : i64
    vm.return
  }

  vm.export @test_and_i64
  vm.func @test_and_i64() {
    %c1 = vm.const.i64 5 : i64
    %c1dno = iree.do_not_optimize(%c1) : i64
    %c2 = vm.const.i64 3 : i64
    %c2dno = iree.do_not_optimize(%c2) : i64
    %v = vm.and.i64 %c1dno, %c2dno : i64
    %c3 = vm.const.i64 1 : i64
    vm.check.eq %v, %c3, "5&3=1" : i64
    vm.return
  }

  vm.export @test_or_i64
  vm.func @test_or_i64() {
    %c1 = vm.const.i64 5 : i64
    %c1dno = iree.do_not_optimize(%c1) : i64
    %c2 = vm.const.i64 3 : i64
    %c2dno = iree.do_not_optimize(%c2) : i64
    %v = vm.or.i64 %c1dno, %c2dno : i64
    %c3 = vm.const.i64 7 : i64
    vm.check.eq %v, %c3, "5|3=7" : i64
    vm.return
  }

  vm.export @test_xor_i64
  vm.func @test_xor_i64() {
    %c1 = vm.const.i64 5 : i64
    %c1dno = iree.do_not_optimize(%c1) : i64
    %c2 = vm.const.i64 3 : i64
    %c2dno = iree.do_not_optimize(%c2) : i64
    %v = vm.xor.i64 %c1dno, %c2dno : i64
    %c3 = vm.const.i64 6 : i64
    vm.check.eq %v, %c3, "5^3=6" : i64
    vm.return
  }
}
