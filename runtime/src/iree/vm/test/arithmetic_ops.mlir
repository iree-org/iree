vm.module @arithmetic_ops {

  //===--------------------------------------------------------------------===//
  // Native integer arithmetic
  //===--------------------------------------------------------------------===//

  vm.export @test_add_i32
  vm.func @test_add_i32() {
    %c1 = vm.const.i32 1
    %c1dno = util.optimization_barrier %c1 : i32
    %v = vm.add.i32 %c1dno, %c1dno : i32
    %c2 = vm.const.i32 2
    vm.check.eq %v, %c2, "1+1=2" : i32
    vm.return
  }

  vm.export @test_sub_i32
  vm.func @test_sub_i32() {
    %c1 = vm.const.i32 3
    %c1dno = util.optimization_barrier %c1 : i32
    %c2 = vm.const.i32 2
    %c2dno = util.optimization_barrier %c2 : i32
    %v = vm.sub.i32 %c1dno, %c2dno : i32
    %c3 = vm.const.i32 1
    vm.check.eq %v, %c3, "3-2=1" : i32
    vm.return
  }

  vm.export @test_mul_i32
  vm.func @test_mul_i32() {
    %c1 = vm.const.i32 2
    %c1dno = util.optimization_barrier %c1 : i32
    %v = vm.mul.i32 %c1dno, %c1dno : i32
    %c2 = vm.const.i32 4
    vm.check.eq %v, %c2, "2*2=4" : i32
    vm.return
  }

  vm.export @test_div_i32s
  vm.func @test_div_i32s() {
    %c1 = vm.const.i32 4
    %c1dno = util.optimization_barrier %c1 : i32
    %c2 = vm.const.i32 -2
    %c2dno = util.optimization_barrier %c2 : i32
    %v = vm.div.i32.s %c1dno, %c2dno : i32
    %c3 = vm.const.i32 -2
    vm.check.eq %v, %c3, "4/-2=-2" : i32
    vm.return
  }

  vm.export @test_div_i32u
  vm.func @test_div_i32u() {
    %c1 = vm.const.i32 4
    %c1dno = util.optimization_barrier %c1 : i32
    %c2 = vm.const.i32 2
    %c2dno = util.optimization_barrier %c2 : i32
    %v = vm.div.i32.u %c1dno, %c2dno : i32
    %c3 = vm.const.i32 2
    vm.check.eq %v, %c3, "4/2=2" : i32
    vm.return
  }

  vm.export @test_rem_i32s
  vm.func @test_rem_i32s() {
    %c1 = vm.const.i32 -3
    %c1dno = util.optimization_barrier %c1 : i32
    %c2 = vm.const.i32 -2
    %c2dno = util.optimization_barrier %c2 : i32
    %v = vm.rem.i32.s %c1dno, %c2dno : i32
    %c3 = vm.const.i32 -1
    vm.check.eq %v, %c3, "-3%-2=-1" : i32
    vm.return
  }

  vm.export @test_rem_i32u
  vm.func @test_rem_i32u() {
    %c1 = vm.const.i32 3
    %c1dno = util.optimization_barrier %c1 : i32
    %c2 = vm.const.i32 2
    %c2dno = util.optimization_barrier %c2 : i32
    %v = vm.rem.i32.u %c1dno, %c2dno : i32
    %c3 = vm.const.i32 1
    vm.check.eq %v, %c3, "3%2=1" : i32
    vm.return
  }

  vm.export @test_fma_i32
  vm.func @test_fma_i32() {
    %c2 = vm.const.i32 2
    %c2dno = util.optimization_barrier %c2 : i32
    %c3 = vm.const.i32 3
    %c3dno = util.optimization_barrier %c3 : i32
    %c5 = vm.const.i32 5
    %c5dno = util.optimization_barrier %c5 : i32
    %v = vm.fma.i32 %c2dno, %c3dno, %c5dno : i32
    %c11 = vm.const.i32 11
    vm.check.eq %v, %c11, "2*3+5=11" : i32
    vm.return
  }

  vm.export @test_abs_i32
  vm.func @test_abs_i32() {
    %cn1 = vm.const.i32 -1
    %cn1dno = util.optimization_barrier %cn1 : i32
    %v = vm.abs.i32 %cn1dno : i32
    %c1 = vm.const.i32 1
    vm.check.eq %v, %c1, "abs(-1)=1" : i32
    vm.return
  }

  vm.export @test_min_i32s
  vm.func @test_min_i32s() {
    %cn3 = vm.const.i32 -3
    %cn3dno = util.optimization_barrier %cn3 : i32
    %c2 = vm.const.i32 2
    %c2dno = util.optimization_barrier %c2 : i32
    %v = vm.min.i32.s %cn3dno, %c2dno : i32
    vm.check.eq %v, %cn3, "smin(-3,2)=-3" : i32
    vm.return
  }

  vm.export @test_min_i32u
  vm.func @test_min_i32u() {
    %cn3 = vm.const.i32 -3
    %cn3dno = util.optimization_barrier %cn3 : i32
    %c2 = vm.const.i32 2
    %c2dno = util.optimization_barrier %c2 : i32
    %v = vm.min.i32.u %cn3dno, %c2dno : i32
    vm.check.eq %v, %c2, "umin(-3,2)=2" : i32
    vm.return
  }

  vm.export @test_max_i32s
  vm.func @test_max_i32s() {
    %cn3 = vm.const.i32 -3
    %cn3dno = util.optimization_barrier %cn3 : i32
    %c2 = vm.const.i32 2
    %c2dno = util.optimization_barrier %c2 : i32
    %v = vm.max.i32.s %cn3dno, %c2dno : i32
    vm.check.eq %v, %c2, "smax(-3,2)=2" : i32
    vm.return
  }

  vm.export @test_max_i32u
  vm.func @test_max_i32u() {
    %cn3 = vm.const.i32 -3
    %cn3dno = util.optimization_barrier %cn3 : i32
    %c2 = vm.const.i32 2
    %c2dno = util.optimization_barrier %c2 : i32
    %v = vm.max.i32.u %cn3dno, %c2dno : i32
    vm.check.eq %v, %cn3, "umax(-3,2)=-3" : i32
    vm.return
  }

  vm.export @test_not_i32
  vm.func @test_not_i32() {
    %c1 = vm.const.i32 0
    %c1dno = util.optimization_barrier %c1 : i32
    %v = vm.not.i32 %c1dno : i32
    %c2 = vm.const.i32 -1
    vm.check.eq %v, %c2, "~0=-1" : i32
    vm.return
  }

  vm.export @test_and_i32
  vm.func @test_and_i32() {
    %c1 = vm.const.i32 5
    %c1dno = util.optimization_barrier %c1 : i32
    %c2 = vm.const.i32 3
    %c2dno = util.optimization_barrier %c2 : i32
    %v = vm.and.i32 %c1dno, %c2dno : i32
    %c3 = vm.const.i32 1
    vm.check.eq %v, %c3, "5&3=1" : i32
    vm.return
  }

  vm.export @test_or_i32
  vm.func @test_or_i32() {
    %c1 = vm.const.i32 5
    %c1dno = util.optimization_barrier %c1 : i32
    %c2 = vm.const.i32 3
    %c2dno = util.optimization_barrier %c2 : i32
    %v = vm.or.i32 %c1dno, %c2dno : i32
    %c3 = vm.const.i32 7
    vm.check.eq %v, %c3, "5|3=7" : i32
    vm.return
  }

  vm.export @test_xor_i32
  vm.func @test_xor_i32() {
    %c1 = vm.const.i32 5
    %c1dno = util.optimization_barrier %c1 : i32
    %c2 = vm.const.i32 3
    %c2dno = util.optimization_barrier %c2 : i32
    %v = vm.xor.i32 %c1dno, %c2dno : i32
    %c3 = vm.const.i32 6
    vm.check.eq %v, %c3, "5^3=6" : i32
    vm.return
  }

  vm.export @test_ctlz_i32_const_zero
  vm.func @test_ctlz_i32_const_zero() {
    %c = vm.const.i32 0
    %cdno = util.optimization_barrier %c : i32
    %actual = vm.ctlz.i32 %cdno : i32
    %expected = vm.const.i32 32
    vm.check.eq %actual, %expected, "ctlz(0)=32" : i32
    vm.return
  }

  vm.export @test_ctlz_i32_const_1
  vm.func @test_ctlz_i32_const_1() {
    %c = vm.const.i32 1
    %cdno = util.optimization_barrier %c : i32
    %actual = vm.ctlz.i32 %cdno : i32
    %expected = vm.const.i32 31
    vm.check.eq %actual, %expected, "ctlz(1)=31" : i32
    vm.return
  }

  vm.export @test_ctlz_i32_const_ffffffff
  vm.func @test_ctlz_i32_const_ffffffff() {
    %c = vm.const.i32 0xFFFFFFFF
    %cdno = util.optimization_barrier %c : i32
    %actual = vm.ctlz.i32 %cdno : i32
    %expected = vm.const.i32 0
    vm.check.eq %actual, %expected, "ctlz(0xFFFFFFFF)=0" : i32
    vm.return
  }
}
