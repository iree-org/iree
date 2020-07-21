vm.module @comparison_ops {

  //===--------------------------------------------------------------------===//
  // vm.cmp.lt.i32.s
  //===--------------------------------------------------------------------===//

  vm.export @test_cmp_lt_s_0
  vm.func @test_cmp_lt_s_0() {
    %lhs = vm.const.i32 2 : i32
    %lhs_dno = iree.do_not_optimize(%lhs) : i32
    %rhs = vm.const.i32 -2 : i32
    %rhs_dno = iree.do_not_optimize(%rhs) : i32
    %actual = vm.cmp.lt.i32.s %lhs_dno, %rhs_dno : i32
    %expected = vm.const.i32 0 : i32
    vm.check.eq %actual, %expected, "2 < -2" : i32
    vm.return
  }

  vm.export @test_cmp_lt_s_1
  vm.func @test_cmp_lt_s_1() {
    %lhs = vm.const.i32 -2 : i32
    %lhs_dno = iree.do_not_optimize(%lhs) : i32
    %rhs = vm.const.i32 2 : i32
    %rhs_dno = iree.do_not_optimize(%rhs) : i32
    %actual = vm.cmp.lt.i32.s %lhs_dno, %rhs_dno : i32
    %expected = vm.const.i32 1 : i32
    vm.check.eq %actual, %expected, "-2 < 2" : i32
    vm.return
  }

  // Expect UINT_MAX to be interpreted as -1 when doing a signed compare.
  vm.export @test_cmp_lt_s_2
  vm.func @test_cmp_lt_s_2() {
    %lhs = vm.const.i32 4294967295 : i32
    %lhs_dno = iree.do_not_optimize(%lhs) : i32
    %rhs = vm.const.i32 2 : i32
    %rhs_dno = iree.do_not_optimize(%rhs) : i32
    %actual = vm.cmp.lt.i32.s %lhs_dno, %rhs_dno : i32
    %expected = vm.const.i32 1 : i32
    vm.check.eq %actual, %expected, "4294967295 (UINT_MAX) < 2" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.cmp.lt.i32.u
  //===--------------------------------------------------------------------===//

  vm.export @test_cmp_lt_u_0
  vm.func @test_cmp_lt_u_0() {
    %lhs = vm.const.i32 2 : i32
    %lhs_dno = iree.do_not_optimize(%lhs) : i32
    %rhs = vm.const.i32 -2 : i32
    %rhs_dno = iree.do_not_optimize(%rhs) : i32
    %actual = vm.cmp.lt.i32.u %lhs_dno, %rhs_dno : i32
    %expected = vm.const.i32 1 : i32
    vm.check.eq %actual, %expected, "2 < -2 (as unsigned)" : i32
    vm.return
  }

  vm.export @test_cmp_lt_u_1
  vm.func @test_cmp_lt_u_1() {
    %lhs = vm.const.i32 -2 : i32
    %lhs_dno = iree.do_not_optimize(%lhs) : i32
    %rhs = vm.const.i32 2 : i32
    %rhs_dno = iree.do_not_optimize(%rhs) : i32
    %actual = vm.cmp.lt.i32.u %lhs_dno, %rhs_dno : i32
    %expected = vm.const.i32 0 : i32
    vm.check.eq %actual, %expected, "-2 < 2 (as unsigned)" : i32
    vm.return
  }

  vm.export @test_cmp_lt_u_2
  vm.func @test_cmp_lt_u_2() {
    %lhs = vm.const.i32 4294967295 : i32
    %lhs_dno = iree.do_not_optimize(%lhs) : i32
    %rhs = vm.const.i32 2 : i32
    %rhs_dno = iree.do_not_optimize(%rhs) : i32
    %actual = vm.cmp.lt.i32.u %lhs_dno, %rhs_dno : i32
    %expected = vm.const.i32 0 : i32
    vm.check.eq %actual, %expected, "4294967295 (UINT_MAX) < 2 (as unsigned)" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.cmp.*.i32.* pseudo-ops
  //===--------------------------------------------------------------------===//
  // NOTE: all of these are turned in to some variants of vm.cmp.lt by the
  // compiler and are here as a way to test the runtime behavior of the
  // pseudo-op expansions.

  vm.export @test_cmp_lte
  vm.func @test_cmp_lte() {
    %true = vm.const.i32 1 : i32
    %false = vm.const.i32 0 : i32

    %cn2 = vm.const.i32 -2 : i32
    %cn2_dno = iree.do_not_optimize(%cn2) : i32
    %c2 = vm.const.i32 2 : i32
    %c2_dno = iree.do_not_optimize(%c2) : i32

    %cmp_0 = vm.cmp.lte.i32.s %cn2_dno, %c2_dno : i32
    vm.check.eq %cmp_0, %true, "-2 <= 2" : i32
    %cmp_1 = vm.cmp.lte.i32.s %c2_dno, %cn2_dno : i32
    vm.check.eq %cmp_1, %false, "2 <= -2" : i32
    %cmp_2 = vm.cmp.lte.i32.s %c2_dno, %c2_dno : i32
    vm.check.eq %cmp_2, %true, "2 <= 2" : i32

    %cmp_3 = vm.cmp.lte.i32.u %cn2_dno, %c2_dno : i32
    vm.check.eq %cmp_3, %false, "-2 <= 2 (unsigned)" : i32
    %cmp_4 = vm.cmp.lte.i32.u %c2_dno, %cn2_dno : i32
    vm.check.eq %cmp_4, %true, "2 <= -2 (unsigned)" : i32
    %cmp_5 = vm.cmp.lte.i32.u %c2_dno, %c2_dno : i32
    vm.check.eq %cmp_5, %true, "2 <= 2 (unsigned)" : i32

    vm.return
  }

  vm.export @test_cmp_gt
  vm.func @test_cmp_gt() {
    %true = vm.const.i32 1 : i32
    %false = vm.const.i32 0 : i32

    %cn2 = vm.const.i32 -2 : i32
    %cn2_dno = iree.do_not_optimize(%cn2) : i32
    %c2 = vm.const.i32 2 : i32
    %c2_dno = iree.do_not_optimize(%c2) : i32

    %cmp_0 = vm.cmp.gt.i32.s %cn2_dno, %c2_dno : i32
    vm.check.eq %cmp_0, %false, "-2 > 2" : i32
    %cmp_1 = vm.cmp.gt.i32.s %c2_dno, %cn2_dno : i32
    vm.check.eq %cmp_1, %true, "2 > -2" : i32
    %cmp_2 = vm.cmp.gt.i32.s %c2_dno, %c2_dno : i32
    vm.check.eq %cmp_2, %false, "2 > 2" : i32

    %cmp_3 = vm.cmp.gt.i32.u %cn2_dno, %c2_dno : i32
    vm.check.eq %cmp_3, %true, "-2 > 2 (unsigned)" : i32
    %cmp_4 = vm.cmp.gt.i32.u %c2_dno, %cn2_dno : i32
    vm.check.eq %cmp_4, %false, "2 > -2 (unsigned)" : i32
    %cmp_5 = vm.cmp.gt.i32.u %c2_dno, %c2_dno : i32
    vm.check.eq %cmp_5, %false, "2 > 2 (unsigned)" : i32

    vm.return
  }

  vm.export @test_cmp_gte
  vm.func @test_cmp_gte() {
    %true = vm.const.i32 1 : i32
    %false = vm.const.i32 0 : i32

    %cn2 = vm.const.i32 -2 : i32
    %cn2_dno = iree.do_not_optimize(%cn2) : i32
    %c2 = vm.const.i32 2 : i32
    %c2_dno = iree.do_not_optimize(%c2) : i32

    %cmp_0 = vm.cmp.gte.i32.s %cn2_dno, %c2_dno : i32
    vm.check.eq %cmp_0, %false, "-2 >= 2" : i32
    %cmp_1 = vm.cmp.gte.i32.s %c2_dno, %cn2_dno : i32
    vm.check.eq %cmp_1, %true, "2 >= -2" : i32
    %cmp_2 = vm.cmp.gte.i32.s %c2_dno, %c2_dno : i32
    vm.check.eq %cmp_2, %true, "2 >= 2" : i32

    %cmp_3 = vm.cmp.gte.i32.u %cn2_dno, %c2_dno : i32
    vm.check.eq %cmp_3, %true, "-2 >= 2 (unsigned)" : i32
    %cmp_4 = vm.cmp.gte.i32.u %c2_dno, %cn2_dno : i32
    vm.check.eq %cmp_4, %false, "2 >= -2 (unsigned)" : i32
    %cmp_5 = vm.cmp.gte.i32.u %c2_dno, %c2_dno : i32
    vm.check.eq %cmp_5, %true, "2 >= 2 (unsigned)" : i32

    vm.return
  }

}
