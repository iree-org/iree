vm.module @arithmetic_ops_f32 {

  //===--------------------------------------------------------------------===//
  // ExtF32: Native floating-point arithmetic
  //===--------------------------------------------------------------------===//

  vm.export @test_add_f32
  vm.func @test_add_f32() {
    %c1 = vm.const.f32 1.5 : f32
    %c1dno = iree.do_not_optimize(%c1) : f32
    %v = vm.add.f32 %c1dno, %c1dno : f32
    %c2 = vm.const.f32 3.0 : f32
    vm.check.eq %v, %c2, "1.5+1.5=3" : f32
    vm.return
  }

  vm.export @test_sub_f32
  vm.func @test_sub_f32() {
    %c1 = vm.const.f32 3.0 : f32
    %c1dno = iree.do_not_optimize(%c1) : f32
    %c2 = vm.const.f32 2.5 : f32
    %c2dno = iree.do_not_optimize(%c2) : f32
    %v = vm.sub.f32 %c1dno, %c2dno : f32
    %c3 = vm.const.f32 0.5 : f32
    vm.check.eq %v, %c3, "3.0-2.5=0.5" : f32
    vm.return
  }

  vm.export @test_mul_f32
  vm.func @test_mul_f32() {
    %c1 = vm.const.f32 2.5 : f32
    %c1dno = iree.do_not_optimize(%c1) : f32
    %v = vm.mul.f32 %c1dno, %c1dno : f32
    %c2 = vm.const.f32 6.25 : f32
    vm.check.eq %v, %c2, "2.5*2.5=6.25" : f32
    vm.return
  }

  vm.export @test_div_f32
  vm.func @test_div_f32() {
    %c1 = vm.const.f32 4.0 : f32
    %c1dno = iree.do_not_optimize(%c1) : f32
    %c2 = vm.const.f32 -2.0 : f32
    %c2dno = iree.do_not_optimize(%c2) : f32
    %v = vm.div.f32 %c1dno, %c2dno : f32
    %c3 = vm.const.f32 -2.0 : f32
    vm.check.eq %v, %c3, "4.0/-2.0=-2.0" : f32
    vm.return
  }

  vm.export @test_rem_f32
  vm.func @test_rem_f32() {
    %c1 = vm.const.f32 -3.0 : f32
    %c1dno = iree.do_not_optimize(%c1) : f32
    %c2 = vm.const.f32 -2.0 : f32
    %c2dno = iree.do_not_optimize(%c2) : f32
    %v = vm.rem.f32 %c1dno, %c2dno : f32
    %c3 = vm.const.f32 1.0 : f32
    vm.check.eq %v, %c3, "-3.0%-2.0=1.0" : f32
    vm.return
  }

  vm.export @test_abs_f32
  vm.func @test_abs_f32() {
    %c1 = vm.const.f32 -1.0 : f32
    %c1dno = iree.do_not_optimize(%c1) : f32
    %v = vm.abs.f32 %c1dno : f32
    %c2 = vm.const.f32 1.0 : f32
    vm.check.eq %v, %c2, "abs(-1.0)=1.0" : f32
    vm.return
  }

  vm.export @test_neg_f32
  vm.func @test_neg_f32() {
    %c1 = vm.const.f32 -1.0 : f32
    %c1dno = iree.do_not_optimize(%c1) : f32
    %v = vm.neg.f32 %c1dno : f32
    %c2 = vm.const.f32 1.0 : f32
    vm.check.eq %v, %c2, "neg(-1.0)=1.0" : f32
    vm.return
  }

  vm.export @test_ceil_f32
  vm.func @test_ceil_f32() {
    %c1 = vm.const.f32 1.5 : f32
    %c1dno = iree.do_not_optimize(%c1) : f32
    %v = vm.ceil.f32 %c1dno : f32
    %c2 = vm.const.f32 2.0 : f32
    vm.check.eq %v, %c2, "ceil(1.5)=2.0" : f32
    vm.return
  }

  vm.export @test_floor_f32
  vm.func @test_floor_f32() {
    %c1 = vm.const.f32 1.5 : f32
    %c1dno = iree.do_not_optimize(%c1) : f32
    %v = vm.floor.f32 %c1dno : f32
    %c2 = vm.const.f32 1.0 : f32
    vm.check.eq %v, %c2, "floor(1.5)=1.0" : f32
    vm.return
  }
}
