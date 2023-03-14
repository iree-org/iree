vm.module @arithmetic_ops_f32 {

  //===--------------------------------------------------------------------===//
  // ExtF32: Native floating-point arithmetic
  //===--------------------------------------------------------------------===//

  vm.export @test_add_f32
  vm.func @test_add_f32() {
    %c1 = vm.const.f32 1.5
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.add.f32 %c1dno, %c1dno : f32
    %c2 = vm.const.f32 3.0
    vm.check.eq %v, %c2, "1.5+1.5=3" : f32
    vm.return
  }

  vm.export @test_sub_f32
  vm.func @test_sub_f32() {
    %c1 = vm.const.f32 3.0
    %c1dno = util.optimization_barrier %c1 : f32
    %c2 = vm.const.f32 2.5
    %c2dno = util.optimization_barrier %c2 : f32
    %v = vm.sub.f32 %c1dno, %c2dno : f32
    %c3 = vm.const.f32 0.5
    vm.check.eq %v, %c3, "3.0-2.5=0.5" : f32
    vm.return
  }

  vm.export @test_mul_f32
  vm.func @test_mul_f32() {
    %c1 = vm.const.f32 2.5
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.mul.f32 %c1dno, %c1dno : f32
    %c2 = vm.const.f32 6.25
    vm.check.eq %v, %c2, "2.5*2.5=6.25" : f32
    vm.return
  }

  vm.export @test_div_f32
  vm.func @test_div_f32() {
    %c1 = vm.const.f32 4.0
    %c1dno = util.optimization_barrier %c1 : f32
    %c2 = vm.const.f32 -2.0
    %c2dno = util.optimization_barrier %c2 : f32
    %v = vm.div.f32 %c1dno, %c2dno : f32
    %c3 = vm.const.f32 -2.0
    vm.check.eq %v, %c3, "4.0/-2.0=-2.0" : f32
    vm.return
  }

  vm.export @test_rem_f32
  vm.func @test_rem_f32() {
    %c1 = vm.const.f32 -3.0
    %c1dno = util.optimization_barrier %c1 : f32
    %c2 = vm.const.f32 -2.0
    %c2dno = util.optimization_barrier %c2 : f32
    %v = vm.rem.f32 %c1dno, %c2dno : f32
    %c3 = vm.const.f32 1.0
    vm.check.eq %v, %c3, "-3.0%-2.0=1.0" : f32
    vm.return
  }

  vm.export @test_fma_f32
  vm.func @test_fma_f32() {
    %c2 = vm.const.f32 2.0
    %c2dno = util.optimization_barrier %c2 : f32
    %c3 = vm.const.f32 3.0
    %c3dno = util.optimization_barrier %c3 : f32
    %c5 = vm.const.f32 5.0
    %c5dno = util.optimization_barrier %c5 : f32
    %v = vm.fma.f32 %c2dno, %c3dno, %c5dno : f32
    %c11 = vm.const.f32 11.0
    vm.check.eq %v, %c11, "2.0*3.0+5.0=11.0" : f32
    vm.return
  }

  vm.export @test_abs_f32
  vm.func @test_abs_f32() {
    %c1 = vm.const.f32 -1.0
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.abs.f32 %c1dno : f32
    %c2 = vm.const.f32 1.0
    vm.check.eq %v, %c2, "abs(-1.0)=1.0" : f32
    vm.return
  }

  vm.export @test_neg_f32
  vm.func @test_neg_f32() {
    %c1 = vm.const.f32 -1.0
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.neg.f32 %c1dno : f32
    %c2 = vm.const.f32 1.0
    vm.check.eq %v, %c2, "neg(-1.0)=1.0" : f32
    vm.return
  }

  vm.export @test_ceil_f32
  vm.func @test_ceil_f32() {
    %c1 = vm.const.f32 1.5
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.ceil.f32 %c1dno : f32
    %c2 = vm.const.f32 2.0
    vm.check.eq %v, %c2, "ceil(1.5)=2.0" : f32
    vm.return
  }

  vm.export @test_floor_f32
  vm.func @test_floor_f32() {
    %c15 = vm.const.f32 1.5
    %c15dno = util.optimization_barrier %c15 : f32
    %v = vm.floor.f32 %c15dno : f32
    %c1 = vm.const.f32 1.0
    vm.check.eq %v, %c1, "floor(1.5)=1.0" : f32
    vm.return
  }

  vm.export @test_round_f32
  vm.func @test_round_f32() {
    %c15 = vm.const.f32 1.5
    %c15dno = util.optimization_barrier %c15 : f32
    %v = vm.round.f32 %c15dno : f32
    %c2 = vm.const.f32 2.0
    vm.check.eq %v, %c2, "round(1.5)=2.0" : f32
    vm.return
  }

  vm.export @test_min_f32
  vm.func @test_min_f32() {
    %cn3 = vm.const.f32 -3.0
    %cn3dno = util.optimization_barrier %cn3 : f32
    %cn2 = vm.const.f32 -2.0
    %cn2dno = util.optimization_barrier %cn2 : f32
    %v = vm.min.f32 %cn3dno, %cn2dno : f32
    vm.check.eq %v, %cn3, "min(-3.0,-2.0)=-3.0" : f32
    vm.return
  }

  vm.export @test_max_f32
  vm.func @test_max_f32() {
    %cn3 = vm.const.f32 -3.0
    %cn3dno = util.optimization_barrier %cn3 : f32
    %cn2 = vm.const.f32 -2.0
    %cn2dno = util.optimization_barrier %cn2 : f32
    %v = vm.max.f32 %cn3dno, %cn2dno : f32
    vm.check.eq %v, %cn2, "max(-3.0,-2.0)=-2.0" : f32
    vm.return
  }

  vm.export @test_atan_f32
  vm.func @test_atan_f32() {
    %c1 = vm.const.f32 1.0
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.atan.f32 %c1dno : f32
    %c2 = vm.const.f32 0.7853981633974483
    vm.check.eq %v, %c2, "atan(1.0)=0.7853981633974483" : f32
    vm.return
  }

  vm.export @test_atan2_f32
  vm.func @test_atan2_f32() {
    %c1 = vm.const.f32 1.0
    %c1dno = util.optimization_barrier %c1 : f32
    %c2 = vm.const.f32 0.0
    %c2dno = util.optimization_barrier %c2 : f32
    %v = vm.atan2.f32 %c1dno, %c2dno : f32
    %c3 = vm.const.f32 1.5707963267948966
    vm.check.eq %v, %c3, "atan2(1.0,0.0)=1.5707963267948966" : f32
    vm.return
  }

  vm.export @test_cos_f32
  vm.func @test_cos_f32() {
    %c1 = vm.const.f32 0.5
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.cos.f32 %c1dno : f32
    %c2 = vm.const.f32 0.8775825618903728
    vm.check.eq %v, %c2, "cos(0.5)=0.8775825618903728" : f32
    vm.return
  }

  vm.export @test_sin_f32
  vm.func @test_sin_f32() {
    %c1 = vm.const.f32 0.5
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.sin.f32 %c1dno : f32
    %c2 = vm.const.f32 0.479425538604203
    vm.check.eq %v, %c2, "sin(0.5)=0.479425538604203" : f32
    vm.return
  }

  vm.export @test_exp_f32
  vm.func @test_exp_f32() {
    %c1 = vm.const.f32 1.0
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.exp.f32 %c1dno : f32
    %c2 = vm.const.f32 2.718281828459045
    vm.check.eq %v, %c2, "exp(1.0)=2.718281828459045" : f32
    vm.return
  }

  vm.export @test_exp2_f32
  vm.func @test_exp2_f32() {
    %c1 = vm.const.f32 2.0
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.exp2.f32 %c1dno : f32
    %c2 = vm.const.f32 4.0
    vm.check.eq %v, %c2, "exp(2.0)=4.0" : f32
    vm.return
  }

  vm.export @test_expm1_f32
  vm.func @test_expm1_f32() {
    %c1 = vm.const.f32 2.0
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.expm1.f32 %c1dno : f32
    %c2 = vm.const.f32 6.38905609893065
    vm.check.eq %v, %c2, "expm1(2.0)=6.38905609893065" : f32
    vm.return
  }

  vm.export @test_log_f32
  vm.func @test_log_f32() {
    %c1 = vm.const.f32 10.0
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.log.f32 %c1dno : f32
    %c2 = vm.const.f32 2.302585092994046
    vm.check.eq %v, %c2, "log(10.0)=2.302585092994046" : f32
    vm.return
  }

  vm.export @test_log10_f32
  vm.func @test_log10_f32() {
    %c1 = vm.const.f32 10.0
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.log10.f32 %c1dno : f32
    %c2 = vm.const.f32 1.0
    vm.check.eq %v, %c2, "log10(10.0)=1.0" : f32
    vm.return
  }

  vm.export @test_log1p_f32
  vm.func @test_log1p_f32() {
    %c1 = vm.const.f32 10.0
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.log1p.f32 %c1dno : f32
    %c2 = vm.const.f32 2.3978952727983707
    vm.check.eq %v, %c2, "log1p(10.0)=2.3978952727983707" : f32
    vm.return
  }

  vm.export @test_log2_f32
  vm.func @test_log2_f32() {
    %c1 = vm.const.f32 10.0
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.log2.f32 %c1dno : f32
    %c2 = vm.const.f32 3.321928094887362
    vm.check.eq %v, %c2, "log2(10.0)=3.321928094887362" : f32
    vm.return
  }

  vm.export @test_pow_f32
  vm.func @test_pow_f32() {
    %c1 = vm.const.f32 3.0
    %c1dno = util.optimization_barrier %c1 : f32
    %c2 = vm.const.f32 2.0
    %c2dno = util.optimization_barrier %c2 : f32
    %v = vm.pow.f32 %c1dno, %c2dno : f32
    %c3 = vm.const.f32 9.0
    vm.check.eq %v, %c3, "pow(3.0,2.0)=9.0" : f32
    vm.return
  }

  vm.export @test_rsqrt_f32
  vm.func @test_rsqrt_f32() {
    %c1 = vm.const.f32 4.0
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.rsqrt.f32 %c1dno : f32
    %c2 = vm.const.f32 0.5
    vm.check.eq %v, %c2, "rsqrt(4.0)=0.5" : f32
    vm.return
  }

  vm.export @test_sqrt_f32
  vm.func @test_sqrt_f32() {
    %c1 = vm.const.f32 4.0
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.sqrt.f32 %c1dno : f32
    %c2 = vm.const.f32 2.0
    vm.check.eq %v, %c2, "sqrt(4.0)=2.0" : f32
    vm.return
  }

  vm.export @test_tanh_f32
  vm.func @test_tanh_f32() {
    %c1 = vm.const.f32 0.5
    %c1dno = util.optimization_barrier %c1 : f32
    %v = vm.tanh.f32 %c1dno : f32
    %c2 = vm.const.f32 0.46211715726000974
    vm.check.eq %v, %c2, "tanh(0.5)=0.46211715726000974" : f32
    vm.return
  }

  // TODO(#5854): vm.check.nearly_eq; this can differ across libm impls.
  // vm.export @test_erf_f32
  // vm.func @test_erf_f32() {
  //   %c1 = vm.const.f32 0.5
  //   %c1dno = util.optimization_barrier %c1 : f32
  //   %v = vm.erf.f32 %c1dno : f32
  //   %c2 = vm.const.f32 0.520499945
  //   vm.check.eq %v, %c2, "erf(0.5)=0.520499945" : f32
  //   vm.return
  // }
}
