vm.module @assignment_ops_i64 {

  //===--------------------------------------------------------------------===//
  // ExtI64: Conditional assignment
  //===--------------------------------------------------------------------===//

  vm.export @test_select_i64
  vm.func @test_select_i64() {
    %c0 = vm.const.i32 0
    %c0dno = util.optimization_barrier %c0 : i32
    %c1 = vm.const.i32 1
    %c1dno = util.optimization_barrier %c1 : i32
    %c2 = vm.const.i64 0
    %c3 = vm.const.i64 1
    %v1 = vm.select.i64 %c0dno, %c2, %c3 : i64
    vm.check.eq %v1, %c3, "0 ? 0 : 1 = 1" : i64
    %v2 = vm.select.i64 %c1dno, %c2, %c3 : i64
    vm.check.eq %v2, %c2, "1 ? 0 : 1 = 0" : i64
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // ExtI64: Lookup table
  //===--------------------------------------------------------------------===//

  vm.export @test_switch_i64 attributes {emitc.exclude}
  vm.func private @test_switch_i64() {
    %c100 = vm.const.i64 100
    %c200 = vm.const.i64 200
    %c300 = vm.const.i64 300

    %i0 = vm.const.i32 0
    %i0_dno = util.optimization_barrier %i0 : i32
    %v0 = vm.switch.i64 %i0_dno[%c100, %c200] else %c300 : i64
    vm.check.eq %v0, %c100, "index 0 is 100" : i64

    %i1 = vm.const.i32 1
    %i1_dno = util.optimization_barrier %i1 : i32
    %v1 = vm.switch.i64 %i1_dno[%c100, %c200] else %c300 : i64
    vm.check.eq %v1, %c200, "index 1 is 200" : i64

    %i2 = vm.const.i32 2
    %i2_dno = util.optimization_barrier %i2 : i32
    %v2 = vm.switch.i64 %i2_dno[%c100, %c200] else %c300 : i64
    vm.check.eq %v2, %c300, "index 2 (out of bounds) is default 300" : i64

    vm.return
  }
}
