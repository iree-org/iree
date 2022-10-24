vm.module @comparison_ops_i64 {

  //===--------------------------------------------------------------------===//
  // vm.cmp.lt.i64.s
  //===--------------------------------------------------------------------===//

  vm.export @test_cmp_lt_s_0_i64
  vm.func @test_cmp_lt_s_0_i64() {
    %lhs = vm.const.i64 4294967295
    %lhs_dno = util.optimization_barrier %lhs : i64
    %rhs = vm.const.i64 -4294967295
    %rhs_dno = util.optimization_barrier %rhs : i64
    %actual = vm.cmp.lt.i64.s %lhs_dno, %rhs_dno : i64
    %expected = vm.const.i32 0
    vm.check.eq %actual, %expected, "4294967295 (UINT_MAX) < -4294967295 (UINT_MAX)" : i32
    vm.return
  }

  vm.export @test_cmp_lt_s_1_i64
  vm.func @test_cmp_lt_s_1_i64() {
    %lhs = vm.const.i64 -4294967295
    %lhs_dno = util.optimization_barrier %lhs : i64
    %rhs = vm.const.i64 4294967295
    %rhs_dno = util.optimization_barrier %rhs : i64
    %actual = vm.cmp.lt.i64.s %lhs_dno, %rhs_dno : i64
    %expected = vm.const.i32 1
    vm.check.eq %actual, %expected, "-4294967295 (UINT_MAX) < 4294967295 (UINT_MAX)" : i32
    vm.return
  }

  // Expect ULONG_MAX to be interpreted as -1 when doing a signed compare.
  vm.export @test_cmp_lt_s_2_i64
  vm.func @test_cmp_lt_s_2_i64() {
    %lhs = vm.const.i64 18446744073709551615
    %lhs_dno = util.optimization_barrier %lhs : i64
    %rhs = vm.const.i64 2
    %rhs_dno = util.optimization_barrier %rhs : i64
    %actual = vm.cmp.lt.i64.s %lhs_dno, %rhs_dno : i64
    %expected = vm.const.i32 1
    vm.check.eq %actual, %expected, "18446744073709551615 (ULONG_MAX) < 2" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.cmp.lt.i64.u
  //===--------------------------------------------------------------------===//

  vm.export @test_cmp_lt_u_0_i64
  vm.func @test_cmp_lt_u_0_i64() {
    %lhs = vm.const.i64 2
    %lhs_dno = util.optimization_barrier %lhs : i64
    %rhs = vm.const.i64 -2
    %rhs_dno = util.optimization_barrier %rhs : i64
    %actual = vm.cmp.lt.i64.u %lhs_dno, %rhs_dno : i64
    %expected = vm.const.i32 1
    vm.check.eq %actual, %expected, "2 < -2 (as unsigned)" : i32
    vm.return
  }

  vm.export @test_cmp_lt_u_1_i64
  vm.func @test_cmp_lt_u_1_i64() {
    %lhs = vm.const.i64 -2
    %lhs_dno = util.optimization_barrier %lhs : i64
    %rhs = vm.const.i64 2
    %rhs_dno = util.optimization_barrier %rhs : i64
    %actual = vm.cmp.lt.i64.u %lhs_dno, %rhs_dno : i64
    %expected = vm.const.i32 0
    vm.check.eq %actual, %expected, "-2 < 2 (as unsigned)" : i32
    vm.return
  }

  vm.export @test_cmp_lt_u_2_i64
  vm.func @test_cmp_lt_u_2_i64() {
    %lhs = vm.const.i64 18446744073709551615
    %lhs_dno = util.optimization_barrier %lhs : i64
    %rhs = vm.const.i64 2
    %rhs_dno = util.optimization_barrier %rhs : i64
    %actual = vm.cmp.lt.i64.u %lhs_dno, %rhs_dno : i64
    %expected = vm.const.i32 0
    vm.check.eq %actual, %expected, "18446744073709551615 (ULONG_MAX) < 2 (as unsigned)" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.cmp.*.i64.* pseudo-ops
  //===--------------------------------------------------------------------===//
  // NOTE: all of these are turned in to some variants of vm.cmp.lt by the
  // compiler and are here as a way to test the runtime behavior of the
  // pseudo-op expansions.

  vm.export @test_cmp_lte_i64
  vm.func @test_cmp_lte_i64() {
    %true = vm.const.i32 1
    %false = vm.const.i32 0

    %cn2 = vm.const.i64 -2
    %cn2_dno = util.optimization_barrier %cn2 : i64
    %c2 = vm.const.i64 2
    %c2_dno = util.optimization_barrier %c2 : i64

    %cmp_0 = vm.cmp.lte.i64.s %cn2_dno, %c2_dno : i64
    vm.check.eq %cmp_0, %true, "-2 <= 2" : i32
    %cmp_1 = vm.cmp.lte.i64.s %c2_dno, %cn2_dno : i64
    vm.check.eq %cmp_1, %false, "2 <= -2" : i32
    %cmp_2 = vm.cmp.lte.i64.s %c2_dno, %c2_dno : i64
    vm.check.eq %cmp_2, %true, "2 <= 2" : i32

    %cmp_3 = vm.cmp.lte.i64.u %cn2_dno, %c2_dno : i64
    vm.check.eq %cmp_3, %false, "-2 <= 2 (unsigned)" : i32
    %cmp_4 = vm.cmp.lte.i64.u %c2_dno, %cn2_dno : i64
    vm.check.eq %cmp_4, %true, "2 <= -2 (unsigned)" : i32
    %cmp_5 = vm.cmp.lte.i64.u %c2_dno, %c2_dno : i64
    vm.check.eq %cmp_5, %true, "2 <= 2 (unsigned)" : i32

    vm.return
  }

  vm.export @test_cmp_gt_i64
  vm.func @test_cmp_gt_i64() {
    %true = vm.const.i32 1
    %false = vm.const.i32 0

    %cn2 = vm.const.i64 -2
    %cn2_dno = util.optimization_barrier %cn2 : i64
    %c2 = vm.const.i64 2
    %c2_dno = util.optimization_barrier %c2 : i64

    %cmp_0 = vm.cmp.gt.i64.s %cn2_dno, %c2_dno : i64
    vm.check.eq %cmp_0, %false, "-2 > 2" : i32
    %cmp_1 = vm.cmp.gt.i64.s %c2_dno, %cn2_dno : i64
    vm.check.eq %cmp_1, %true, "2 > -2" : i32
    %cmp_2 = vm.cmp.gt.i64.s %c2_dno, %c2_dno : i64
    vm.check.eq %cmp_2, %false, "2 > 2" : i32

    %cmp_3 = vm.cmp.gt.i64.u %cn2_dno, %c2_dno : i64
    vm.check.eq %cmp_3, %true, "-2 > 2 (unsigned)" : i32
    %cmp_4 = vm.cmp.gt.i64.u %c2_dno, %cn2_dno : i64
    vm.check.eq %cmp_4, %false, "2 > -2 (unsigned)" : i32
    %cmp_5 = vm.cmp.gt.i64.u %c2_dno, %c2_dno : i64
    vm.check.eq %cmp_5, %false, "2 > 2 (unsigned)" : i32

    vm.return
  }

  vm.export @test_cmp_gte_i64
  vm.func @test_cmp_gte_i64() {
    %true = vm.const.i32 1
    %false = vm.const.i32 0

    %cn2 = vm.const.i64 -2
    %cn2_dno = util.optimization_barrier %cn2 : i64
    %c2 = vm.const.i64 2
    %c2_dno = util.optimization_barrier %c2 : i64

    %cmp_0 = vm.cmp.gte.i64.s %cn2_dno, %c2_dno : i64
    vm.check.eq %cmp_0, %false, "-2 >= 2" : i32
    %cmp_1 = vm.cmp.gte.i64.s %c2_dno, %cn2_dno : i64
    vm.check.eq %cmp_1, %true, "2 >= -2" : i32
    %cmp_2 = vm.cmp.gte.i64.s %c2_dno, %c2_dno : i64
    vm.check.eq %cmp_2, %true, "2 >= 2" : i32

    %cmp_3 = vm.cmp.gte.i64.u %cn2_dno, %c2_dno : i64
    vm.check.eq %cmp_3, %true, "-2 >= 2 (unsigned)" : i32
    %cmp_4 = vm.cmp.gte.i64.u %c2_dno, %cn2_dno : i64
    vm.check.eq %cmp_4, %false, "2 >= -2 (unsigned)" : i32
    %cmp_5 = vm.cmp.gte.i64.u %c2_dno, %c2_dno : i64
    vm.check.eq %cmp_5, %true, "2 >= 2 (unsigned)" : i32

    vm.return
  }
}
