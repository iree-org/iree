vm.module @arithmetic_ops {

  //===--------------------------------------------------------------------===//
  // Native integer arithmetic
  //===--------------------------------------------------------------------===//

  // TODO: The CModuleTarget enforces exports to be ordered.
  vm.export @test_add_i32
  vm.export @test_and_i32
  vm.export @test_div_i32s
  vm.export @test_div_i32u
  vm.export @test_mul_i32
  vm.export @test_not_i32
  vm.export @test_or_i32
  vm.export @test_rem_i32s
  vm.export @test_rem_i32u
  vm.export @test_sub_i32
  vm.export @test_xor_i32

  vm.func @test_add_i32() {
    %c1 = vm.const.i32 1 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %v = vm.add.i32 %c1dno, %c1dno : i32
    %c2 = vm.const.i32 2 : i32
    vm.check.eq %v, %c2, "1+1=2" : i32
    vm.return
  }

  vm.func @test_sub_i32() {
    %c1 = vm.const.i32 3 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %c2 = vm.const.i32 2 : i32
    %c2dno = iree.do_not_optimize(%c2) : i32
    %v = vm.sub.i32 %c1dno, %c2dno : i32
    %c3 = vm.const.i32 1 : i32
    vm.check.eq %v, %c3, "3-2=1" : i32
    vm.return
  }

  vm.func @test_mul_i32() {
    %c1 = vm.const.i32 2 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %v = vm.mul.i32 %c1dno, %c1dno : i32
    %c2 = vm.const.i32 4 : i32
    vm.check.eq %v, %c2, "2*2=4" : i32
    vm.return
  }

  vm.func @test_div_i32s() {
    %c1 = vm.const.i32 4 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %c2 = vm.const.i32 -2 : i32
    %c2dno = iree.do_not_optimize(%c2) : i32
    %v = vm.div.i32.s %c1dno, %c2dno : i32
    %c3 = vm.const.i32 -2 : i32
    vm.check.eq %v, %c3, "4/-2=-2" : i32
    vm.return
  }

  vm.func @test_div_i32u() {
    %c1 = vm.const.i32 4 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %c2 = vm.const.i32 2 : i32
    %c2dno = iree.do_not_optimize(%c2) : i32
    %v = vm.div.i32.u %c1dno, %c2dno : i32
    %c3 = vm.const.i32 2 : i32
    vm.check.eq %v, %c3, "4/2=2" : i32
    vm.return
  }

  vm.func @test_rem_i32s() {
    %c1 = vm.const.i32 -3 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %c2 = vm.const.i32 -2 : i32
    %c2dno = iree.do_not_optimize(%c2) : i32
    %v = vm.rem.i32.s %c1dno, %c2dno : i32
    %c3 = vm.const.i32 -1 : i32
    vm.check.eq %v, %c3, "-3%-2=-1" : i32
    vm.return
  }

  vm.func @test_rem_i32u() {
    %c1 = vm.const.i32 3 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %c2 = vm.const.i32 2 : i32
    %c2dno = iree.do_not_optimize(%c2) : i32
    %v = vm.rem.i32.u %c1dno, %c2dno : i32
    %c3 = vm.const.i32 1 : i32
    vm.check.eq %v, %c3, "3%2=1" : i32
    vm.return
  }

  vm.func @test_not_i32() {
    %c1 = vm.const.i32 0 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %v = vm.not.i32 %c1dno : i32
    %c2 = vm.const.i32 -1 : i32
    vm.check.eq %v, %c2, "~0=-1" : i32
    vm.return
  }

  vm.func @test_and_i32() {
    %c1 = vm.const.i32 5 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %c2 = vm.const.i32 3 : i32
    %c2dno = iree.do_not_optimize(%c2) : i32
    %v = vm.and.i32 %c1dno, %c2dno : i32
    %c3 = vm.const.i32 1 : i32
    vm.check.eq %v, %c3, "5&3=1" : i32
    vm.return
  }

  vm.func @test_or_i32() {
    %c1 = vm.const.i32 5 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %c2 = vm.const.i32 3 : i32
    %c2dno = iree.do_not_optimize(%c2) : i32
    %v = vm.or.i32 %c1dno, %c2dno : i32
    %c3 = vm.const.i32 7 : i32
    vm.check.eq %v, %c3, "5|3=7" : i32
    vm.return
  }

  vm.func @test_xor_i32() {
    %c1 = vm.const.i32 5 : i32
    %c1dno = iree.do_not_optimize(%c1) : i32
    %c2 = vm.const.i32 3 : i32
    %c2dno = iree.do_not_optimize(%c2) : i32
    %v = vm.xor.i32 %c1dno, %c2dno : i32
    %c3 = vm.const.i32 6 : i32
    vm.check.eq %v, %c3, "5^3=6" : i32
    vm.return
  }
}
