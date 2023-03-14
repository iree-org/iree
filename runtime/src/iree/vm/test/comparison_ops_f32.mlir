vm.module @comparison_ops_f32 {

  //===--------------------------------------------------------------------===//
  // vm.cmp.lt.f32
  //===--------------------------------------------------------------------===//

  vm.export @test_cmp_lt_0_f32
  vm.func @test_cmp_lt_0_f32() {
    %lhs = vm.const.f32 4.0
    %lhs_dno = util.optimization_barrier %lhs : f32
    %rhs = vm.const.f32 -4.0
    %rhs_dno = util.optimization_barrier %rhs : f32
    %actual = vm.cmp.lt.f32.o %lhs_dno, %rhs_dno : f32
    %expected = vm.const.i32 0
    vm.check.eq %actual, %expected, "4.0 < -4.0" : i32
    vm.return
  }

  vm.export @test_cmp_lt_1_f32
  vm.func @test_cmp_lt_1_f32() {
    %lhs = vm.const.f32 -4.0
    %lhs_dno = util.optimization_barrier %lhs : f32
    %rhs = vm.const.f32 4.0
    %rhs_dno = util.optimization_barrier %rhs : f32
    %actual = vm.cmp.lt.f32.o %lhs_dno, %rhs_dno : f32
    %expected = vm.const.i32 1
    vm.check.eq %actual, %expected, "-4.0 < 4.0" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.cmp.*.f32 pseudo-ops
  //===--------------------------------------------------------------------===//
  // NOTE: all of these are turned in to some variants of vm.cmp.* and other
  // ops by the compiler and are here as a way to test the runtime behavior of
  // the pseudo-op expansions.

  vm.export @test_cmp_eq_f32_near
  vm.func @test_cmp_eq_f32_near() {
    %true = vm.const.i32 1
    %false = vm.const.i32 0

    %cn2 = vm.const.f32 -2.0
    %cn2_dno = util.optimization_barrier %cn2 : f32
    %c2 = vm.const.f32 2.0
    %c2_dno = util.optimization_barrier %c2 : f32

    %cmp_0 = vm.cmp.eq.f32.near %cn2_dno, %c2_dno : f32
    vm.check.eq %cmp_0, %false, "-2 !~ 2" : i32
    %cmp_1 = vm.cmp.eq.f32.near %c2_dno, %cn2_dno : f32
    vm.check.eq %cmp_1, %false, "2 !~ -2" : i32
    %cmp_2 = vm.cmp.eq.f32.near %c2_dno, %c2_dno : f32
    vm.check.eq %cmp_2, %true, "2 ~ 2" : i32
    %cmp_3 = vm.cmp.eq.f32.near %cn2_dno, %cn2_dno : f32
    vm.check.eq %cmp_3, %true, "-2 ~ -2" : i32

    // off by 84 ULPs, arbitrary threshold sets these as "near enough"
    %c1a = vm.const.f32 1.00002
    %c1a_dno = util.optimization_barrier %c1a : f32
    %c1b = vm.const.f32 1.00003
    %c1b_dno = util.optimization_barrier %c1b : f32

    %cmp_4 = vm.cmp.eq.f32.near %c1a_dno, %c1b_dno : f32
    vm.check.eq %cmp_4, %true, "1.00002 ~ 1.00003" : i32
    %cmp_5 = vm.cmp.eq.f32.near %c1a_dno, %c2_dno : f32
    vm.check.eq %cmp_5, %false, "1.00002 !~ 2" : i32

    vm.return
  }

  vm.export @test_cmp_lte_f32
  vm.func @test_cmp_lte_f32() {
    %true = vm.const.i32 1
    %false = vm.const.i32 0

    %cn2 = vm.const.f32 -2.0
    %cn2_dno = util.optimization_barrier %cn2 : f32
    %c2 = vm.const.f32 2.0
    %c2_dno = util.optimization_barrier %c2 : f32

    %cmp_0 = vm.cmp.lte.f32.o %cn2_dno, %c2_dno : f32
    vm.check.eq %cmp_0, %true, "-2 <= 2" : i32
    %cmp_1 = vm.cmp.lte.f32.o %c2_dno, %cn2_dno : f32
    vm.check.eq %cmp_1, %false, "2 <= -2" : i32
    %cmp_2 = vm.cmp.lte.f32.o %c2_dno, %c2_dno : f32
    vm.check.eq %cmp_2, %true, "2 <= 2" : i32

    vm.return
  }

  vm.export @test_cmp_gt_f32
  vm.func @test_cmp_gt_f32() {
    %true = vm.const.i32 1
    %false = vm.const.i32 0

    %cn2 = vm.const.f32 -2.0
    %cn2_dno = util.optimization_barrier %cn2 : f32
    %c2 = vm.const.f32 2.0
    %c2_dno = util.optimization_barrier %c2 : f32

    %cmp_0 = vm.cmp.gt.f32.o %cn2_dno, %c2_dno : f32
    vm.check.eq %cmp_0, %false, "-2 > 2" : i32
    %cmp_1 = vm.cmp.gt.f32.o %c2_dno, %cn2_dno : f32
    vm.check.eq %cmp_1, %true, "2 > -2" : i32
    %cmp_2 = vm.cmp.gt.f32.o %c2_dno, %c2_dno : f32
    vm.check.eq %cmp_2, %false, "2 > 2" : i32

    vm.return
  }

  vm.export @test_cmp_gte_f32
  vm.func @test_cmp_gte_f32() {
    %true = vm.const.i32 1
    %false = vm.const.i32 0

    %cn2 = vm.const.f32 -2.0
    %cn2_dno = util.optimization_barrier %cn2 : f32
    %c2 = vm.const.f32 2.0
    %c2_dno = util.optimization_barrier %c2 : f32

    %cmp_0 = vm.cmp.gte.f32.o %cn2_dno, %c2_dno : f32
    vm.check.eq %cmp_0, %false, "-2 >= 2" : i32
    %cmp_1 = vm.cmp.gte.f32.o %c2_dno, %cn2_dno : f32
    vm.check.eq %cmp_1, %true, "2 >= -2" : i32
    %cmp_2 = vm.cmp.gte.f32.o %c2_dno, %c2_dno : f32
    vm.check.eq %cmp_2, %true, "2 >= 2" : i32

    vm.return
  }
}
