vm.module @arithmetic_ops_f64 {

  //===--------------------------------------------------------------------===//
  // ExtF64: Native floating-point arithmetic
  //===--------------------------------------------------------------------===//

  vm.export @test_add_f64
  vm.func @test_add_f64() {
    %c1 = vm.const.f64 1.5
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.add.f64 %c1dno, %c1dno : f64
    %c2 = vm.const.f64 3.0
    vm.check.eq %v, %c2, "1.5+1.5=3" : f64
    vm.return
  }

  vm.export @test_sub_f64
  vm.func @test_sub_f64() {
    %c1 = vm.const.f64 3.0
    %c1dno = util.optimization_barrier %c1 : f64
    %c2 = vm.const.f64 2.5
    %c2dno = util.optimization_barrier %c2 : f64
    %v = vm.sub.f64 %c1dno, %c2dno : f64
    %c3 = vm.const.f64 0.5
    vm.check.eq %v, %c3, "3.0-2.5=0.5" : f64
    vm.return
  }

  vm.export @test_mul_f64
  vm.func @test_mul_f64() {
    %c1 = vm.const.f64 2.5
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.mul.f64 %c1dno, %c1dno : f64
    %c2 = vm.const.f64 6.25
    vm.check.eq %v, %c2, "2.5*2.5=6.25" : f64
    vm.return
  }

  vm.export @test_div_f64
  vm.func @test_div_f64() {
    %c1 = vm.const.f64 4.0
    %c1dno = util.optimization_barrier %c1 : f64
    %c2 = vm.const.f64 -2.0
    %c2dno = util.optimization_barrier %c2 : f64
    %v = vm.div.f64 %c1dno, %c2dno : f64
    %c3 = vm.const.f64 -2.0
    vm.check.eq %v, %c3, "4.0/-2.0=-2.0" : f64
    vm.return
  }

  vm.export @test_rem_f64
  vm.func @test_rem_f64() {
    %c1 = vm.const.f64 -3.0
    %c1dno = util.optimization_barrier %c1 : f64
    %c2 = vm.const.f64 -2.0
    %c2dno = util.optimization_barrier %c2 : f64
    %v = vm.rem.f64 %c1dno, %c2dno : f64
    %c3 = vm.const.f64 1.0
    vm.check.eq %v, %c3, "-3.0%-2.0=1.0" : f64
    vm.return
  }

  vm.export @test_fma_f64
  vm.func @test_fma_f64() {
    %c2 = vm.const.f64 2.0
    %c2dno = util.optimization_barrier %c2 : f64
    %c3 = vm.const.f64 3.0
    %c3dno = util.optimization_barrier %c3 : f64
    %c5 = vm.const.f64 5.0
    %c5dno = util.optimization_barrier %c5 : f64
    %v = vm.fma.f64 %c2dno, %c3dno, %c5dno : f64
    %c11 = vm.const.f64 11.0
    vm.check.eq %v, %c11, "2.0*3.0+5.0=11.0" : f64
    vm.return
  }

  vm.export @test_abs_f64
  vm.func @test_abs_f64() {
    %c1 = vm.const.f64 -1.0
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.abs.f64 %c1dno : f64
    %c2 = vm.const.f64 1.0
    vm.check.eq %v, %c2, "abs(-1.0)=1.0" : f64
    vm.return
  }

  vm.export @test_neg_f64
  vm.func @test_neg_f64() {
    %c1 = vm.const.f64 -1.0
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.neg.f64 %c1dno : f64
    %c2 = vm.const.f64 1.0
    vm.check.eq %v, %c2, "neg(-1.0)=1.0" : f64
    vm.return
  }

  vm.export @test_ceil_f64
  vm.func @test_ceil_f64() {
    %c1 = vm.const.f64 1.5
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.ceil.f64 %c1dno : f64
    %c2 = vm.const.f64 2.0
    vm.check.eq %v, %c2, "ceil(1.5)=2.0" : f64
    vm.return
  }

  vm.export @test_floor_f64
  vm.func @test_floor_f64() {
    %c15 = vm.const.f64 1.5
    %c15dno = util.optimization_barrier %c15 : f64
    %v = vm.floor.f64 %c15dno : f64
    %c1 = vm.const.f64 1.0
    vm.check.eq %v, %c1, "floor(1.5)=1.0" : f64
    vm.return
  }

  vm.export @test_round_f64
  vm.func @test_round_f64() {
    %c15 = vm.const.f64 1.5
    %c15dno = util.optimization_barrier %c15 : f64
    %v = vm.round.f64 %c15dno : f64
    %c2 = vm.const.f64 2.0
    vm.check.eq %v, %c2, "round(1.5)=2.0" : f64
    vm.return
  }

  vm.export @test_round_f64_even
  vm.func @test_round_f64_even() {
    %c15 = vm.const.f64 1.5
    %c15dno = util.optimization_barrier %c15 : f64
    %v = vm.round.f64.even %c15dno : f64
    %c2 = vm.const.f64 2.0
    vm.check.eq %v, %c2, "roundeven(1.5)=2.0" : f64
    vm.return
  }

  vm.export @test_min_f64
  vm.func @test_min_f64() {
    %cn3 = vm.const.f64 -3.0
    %cn3dno = util.optimization_barrier %cn3 : f64
    %cn2 = vm.const.f64 -2.0
    %cn2dno = util.optimization_barrier %cn2 : f64
    %v = vm.min.f64 %cn3dno, %cn2dno : f64
    vm.check.eq %v, %cn3, "min(-3.0,-2.0)=-3.0" : f64
    vm.return
  }

  vm.export @test_max_f64
  vm.func @test_max_f64() {
    %cn3 = vm.const.f64 -3.0
    %cn3dno = util.optimization_barrier %cn3 : f64
    %cn2 = vm.const.f64 -2.0
    %cn2dno = util.optimization_barrier %cn2 : f64
    %v = vm.max.f64 %cn3dno, %cn2dno : f64
    vm.check.eq %v, %cn2, "max(-3.0,-2.0)=-2.0" : f64
    vm.return
  }

  vm.export @test_atan_f64
  vm.func @test_atan_f64() {
    %c1 = vm.const.f64 1.0
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.atan.f64 %c1dno : f64
    %c2 = vm.const.f64 0.7853981633974483
    vm.check.eq %v, %c2, "atan(1.0)=0.7853981633974483" : f64
    vm.return
  }

  vm.export @test_atan2_f64
  vm.func @test_atan2_f64() {
    %c1 = vm.const.f64 1.0
    %c1dno = util.optimization_barrier %c1 : f64
    %c2 = vm.const.f64 0.0
    %c2dno = util.optimization_barrier %c2 : f64
    %v = vm.atan2.f64 %c1dno, %c2dno : f64
    %c3 = vm.const.f64 1.5707963267948966
    vm.check.eq %v, %c3, "atan2(1.0,0.0)=1.5707963267948966" : f64
    vm.return
  }

  vm.export @test_cos_f64
  vm.func @test_cos_f64() {
    %c1 = vm.const.f64 0.5
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.cos.f64 %c1dno : f64
    %c2 = vm.const.f64 0.8775825618903728
    vm.check.eq %v, %c2, "cos(0.5)=0.8775825618903728" : f64
    vm.return
  }

  vm.export @test_sin_f64
  vm.func @test_sin_f64() {
    %c1 = vm.const.f64 0.5
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.sin.f64 %c1dno : f64
    %c2 = vm.const.f64 0.479425538604203
    vm.check.eq %v, %c2, "sin(0.5)=0.479425538604203" : f64
    vm.return
  }

  vm.export @test_exp_f64
  vm.func @test_exp_f64() {
    %c1 = vm.const.f64 1.0
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.exp.f64 %c1dno : f64
    %c2 = vm.const.f64 2.718281828459045
    vm.check.eq %v, %c2, "exp(1.0)=2.718281828459045" : f64
    vm.return
  }

  vm.export @test_exp2_f64
  vm.func @test_exp2_f64() {
    %c1 = vm.const.f64 2.0
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.exp2.f64 %c1dno : f64
    %c2 = vm.const.f64 4.0
    vm.check.eq %v, %c2, "exp(2.0)=4.0" : f64
    vm.return
  }

  vm.export @test_expm1_f64
  vm.func @test_expm1_f64() {
    %c1 = vm.const.f64 2.0
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.expm1.f64 %c1dno : f64
    %c2 = vm.const.f64 6.38905609893065
    vm.check.eq %v, %c2, "expm1(2.0)=6.38905609893065" : f64
    vm.return
  }

  vm.export @test_log_f64
  vm.func @test_log_f64() {
    %c1 = vm.const.f64 10.0
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.log.f64 %c1dno : f64
    %c2 = vm.const.f64 2.302585092994046
    vm.check.eq %v, %c2, "log(10.0)=2.302585092994046" : f64
    vm.return
  }

  vm.export @test_log10_f64
  vm.func @test_log10_f64() {
    %c1 = vm.const.f64 10.0
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.log10.f64 %c1dno : f64
    %c2 = vm.const.f64 1.0
    vm.check.eq %v, %c2, "log10(10.0)=1.0" : f64
    vm.return
  }

  vm.export @test_log1p_f64
  vm.func @test_log1p_f64() {
    %c1 = vm.const.f64 10.0
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.log1p.f64 %c1dno : f64
    %c2 = vm.const.f64 2.3978952727983707
    vm.check.eq %v, %c2, "log1p(10.0)=2.3978952727983707" : f64
    vm.return
  }

  vm.export @test_log2_f64
  vm.func @test_log2_f64() {
    %c1 = vm.const.f64 10.0
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.log2.f64 %c1dno : f64
    %c2 = vm.const.f64 3.321928094887362
    vm.check.eq %v, %c2, "log2(10.0)=3.321928094887362" : f64
    vm.return
  }

  vm.export @test_pow_f64
  vm.func @test_pow_f64() {
    %c1 = vm.const.f64 3.0
    %c1dno = util.optimization_barrier %c1 : f64
    %c2 = vm.const.f64 2.0
    %c2dno = util.optimization_barrier %c2 : f64
    %v = vm.pow.f64 %c1dno, %c2dno : f64
    %c3 = vm.const.f64 9.0
    vm.check.eq %v, %c3, "pow(3.0,2.0)=9.0" : f64
    vm.return
  }

  vm.export @test_rsqrt_f64
  vm.func @test_rsqrt_f64() {
    %c1 = vm.const.f64 4.0
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.rsqrt.f64 %c1dno : f64
    %c2 = vm.const.f64 0.5
    vm.check.eq %v, %c2, "rsqrt(4.0)=0.5" : f64
    vm.return
  }

  vm.export @test_sqrt_f64
  vm.func @test_sqrt_f64() {
    %c1 = vm.const.f64 4.0
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.sqrt.f64 %c1dno : f64
    %c2 = vm.const.f64 2.0
    vm.check.eq %v, %c2, "sqrt(4.0)=2.0" : f64
    vm.return
  }

  vm.export @test_tanh_f64
  vm.func @test_tanh_f64() {
    %c1 = vm.const.f64 0.5
    %c1dno = util.optimization_barrier %c1 : f64
    %v = vm.tanh.f64 %c1dno : f64
    %c2 = vm.const.f64 0.46211715726000974
    vm.check.eq %v, %c2, "tanh(0.5)=0.46211715726000974" : f64
    vm.return
  }

  // TODO(#5854): vm.check.nearly_eq; this can differ across libm impls.
  // vm.export @test_erf_f64
  // vm.func @test_erf_f64() {
  //   %c1 = vm.const.f64 0.5
  //   %c1dno = util.optimization_barrier %c1 : f64
  //   %v = vm.erf.f64 %c1dno : f64
  //   %c2 = vm.const.f64 0.520499945
  //   vm.check.eq %v, %c2, "erf(0.5)=0.520499945" : f64
  //   vm.return
  // }
}
